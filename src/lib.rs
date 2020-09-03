//! This crate implements a lockfree cuckoo hashmap.
#![deny(
    // The following are allowed by default lints according to
    // https://doc.rust-lang.org/rustc/lints/listing/allowed-by-default.html
    anonymous_parameters,
    bare_trait_objects,
    // box_pointers, // futures involve boxed pointers
    elided_lifetimes_in_paths, // allow anonymous lifetime in generated code
    missing_copy_implementations,
    missing_debug_implementations,
    // missing_docs, // TODO: add documents
    single_use_lifetimes, // TODO: fix lifetime names only used once
    trivial_casts, // TODO: remove trivial casts in code
    trivial_numeric_casts,
    // unreachable_pub, use clippy::redundant_pub_crate instead
    // unsafe_code, unsafe codes are inevitable here
    // unstable_features,
    unused_extern_crates,
    unused_import_braces,
    unused_qualifications,
    // unused_results, // TODO: fix unused results
    variant_size_differences,

    // Treat warnings as errors
    // warnings, TODO: treat all wanings as errors

    clippy::all,
    clippy::restriction,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo
)]
#![allow(
    // Some explicitly allowed Clippy lints, must have clear reason to allow
    clippy::implicit_return, // actually omitting the return keyword is idiomatic Rust code
    clippy::indexing_slicing,
    clippy::integer_arithmetic,
)]

/// `pointer` defines atomic pointers which will be used for lockfree operations.
mod pointer;

use std::collections::hash_map::RandomState;

use std::convert::TryFrom;

use std::hash::{BuildHasher, Hash, Hasher};

use std::sync::atomic::{AtomicUsize, Ordering};

use pointer::{AtomicPtr, SharedPtr};

use crossbeam_epoch::Owned;

// Re-export `crossbeam_epoch::pin()` and `crossbeam_epoch::Guard`.
pub use crossbeam_epoch::{pin, Guard};

/// `KVPair` contains the key-value pair.
#[derive(Debug)]
#[allow(clippy::missing_docs_in_private_items)]
struct KVPair<K, V> {
    // TODO: maybe cache both hash keys here.
    key: K,
    value: V,
}

/// `SlotIndex` represents the index of a slot inside the hashtable.
/// The slot index is composed by `tbl_idx` and `slot_idx`.
#[derive(Clone, Copy, Debug)]
struct SlotIndex {
    /// `tbl_idx` is the index of the table.
    tbl_idx: usize,
    /// `slot_idx` is the index of the slot inside one table.
    slot_idx: usize,
}

/// `SlotState` represents the state of a slot.
/// A slot could be in one of the four states: null, key, reloc and copied.
enum SlotState {
    /// `NullOrKey` means a slot is empty(null) or is ocupied by a key-value
    /// pair normally without any other flags.
    NullOrKey,
    /// `Reloc` means a slot is being relocated to the other slot.
    Reloc,
    /// `Copied` means a slot is being copied to the new map during resize or
    /// has been copied to the new map.
    Copied,
}

impl SlotState {
    /// `as_u8` converts a `SlotState` to `u8`.
    fn as_u8(&self) -> u8 {
        match self {
            Self::NullOrKey => 0,
            Self::Reloc => 1,
            Self::Copied => 2,
        }
    }

    /// `from_u8` converts a `u8` to `SlotState`.
    fn from_u8(state: u8) -> Self {
        match state {
            0 => Self::NullOrKey,
            1 => Self::Reloc,
            2 => Self::Copied,
            _ => panic!("Invalid slot state from u8: {}", state),
        }
    }
}

/// `MapInner` is the inner implementation of the `LockFreeCuckooHash`.
struct MapInner<K, V> {
    // TODO: support customized hasher.
    /// `hash_builders` is used to hash the keys.
    hash_builders: [RandomState; 2],
    /// `tables` contains the key-value pairs.
    tables: Vec<Vec<AtomicPtr<KVPair<K, V>>>>,
    /// `size` is the number of inserted pairs of the hash map.
    size: AtomicUsize,
    // TODO: For resize
    // copy_batch_num: AtomicUsize,
    // copied_num: AtomicUsize,
    // new_map: AtomicPtr<MapInner<K, V>>,
}

impl<K, V> std::fmt::Debug for MapInner<K, V>
where
    K: std::fmt::Debug,
    V: std::fmt::Debug,
{
    // This is not thread-safe.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let capacity = self.tables[0].len();
        let guard = pin();
        let mut f = f.debug_map();
        for tbl_idx in 0..2 {
            for slot_idx in 0..capacity {
                let slot = self.tables[tbl_idx][slot_idx].load(Ordering::SeqCst, &guard);
                unsafe {
                    if let Some(kv) = slot.as_raw().as_ref() {
                        f.entry(&kv.key, &kv.value);
                    }
                }
            }
        }
        f.finish()
    }
}

impl<'guard, K, V> MapInner<K, V>
where
    K: 'guard + Eq + Hash,
{
    /// `with_capacity` creates a new `MapInner` with specified capacity.
    #[allow(clippy::integer_division)]
    fn with_capacity(capacity: usize, hash_builders: [RandomState; 2]) -> Self {
        let table_capacity = (capacity + 1) / 2;
        let mut tables = Vec::with_capacity(2);

        for _ in 0..2 {
            let mut table = Vec::with_capacity(table_capacity);
            for _ in 0..table_capacity {
                table.push(AtomicPtr::null());
            }
            tables.push(table);
        }

        Self {
            hash_builders,
            tables,
            size: AtomicUsize::new(0),
            // copy_batch_num: AtomicUsize::new(0),
            // copied_num: AtomicUsize::new(0),
            // new_map: AtomicPtr::null(),
        }
    }

    /// `capacity` returns the current capacity of the hash map.
    fn capacity(&self) -> usize {
        self.tables[0].len() * 2
    }

    /// `size` returns the number of inserted pairs of the hash map.
    fn size(&self) -> usize {
        self.size.load(Ordering::SeqCst)
    }

    /// `search` searches the value corresponding to the key.
    fn search(&self, key: &K, guard: &'guard Guard) -> Option<&'guard V> {
        // TODO: K could be a Borrowed.
        let slot_idx0 = self.get_index(0, key);
        // TODO: the second hash value could be lazily evaluated.
        let slot_idx1 = self.get_index(1, key);

        // Because other concurrent `insert` operations may relocate the key during
        // our `search` here, we may miss the key with one-round query.
        // For example, suppose the key is located in `table[1][hash1(key)]` at first:
        //
        //      search thread              |    relocate thread
        //                                 |
        //   e1 = table[0][hash0(key)]     |
        //                                 | relocate key from table[1] to table[0]
        //   e2 = table[1][hash1(key)]     |
        //                                 |
        //   both e1 and e2 are empty      |
        // -> key not exists, return None  |

        // So `search` uses a two-round query to deal with the `missing key` problem.
        // But it is not enough because a relocation operation might interleave in between.
        // The other technique to deal with it is a logic-clock based counter -- `relocation count`.
        // Each slot contains a counter that records the number of relocations at the slot.
        loop {
            // The first round:
            let (count0_0, entry0, _) = self.get_entry(slot_idx0, guard);
            if let Some(pair) = entry0 {
                if pair.key.eq(key) {
                    return Some(&pair.value);
                }
            }

            let (count0_1, entry1, _) = self.get_entry(slot_idx1, guard);
            if let Some(pair) = entry1 {
                if pair.key.eq(key) {
                    return Some(&pair.value);
                }
            }

            // The second round:
            let (count1_0, entry0, _) = self.get_entry(slot_idx0, guard);
            if let Some(pair) = entry0 {
                if pair.key.eq(key) {
                    return Some(&pair.value);
                }
            }

            let (count1_1, entry1, _) = self.get_entry(slot_idx1, guard);
            if let Some(pair) = entry1 {
                if pair.key.eq(key) {
                    return Some(&pair.value);
                }
            }

            // Check the counter.
            if Self::check_counter(count0_0, count0_1, count1_0, count1_1) {
                continue;
            }
            break;
        }
        None
    }

    /// Insert a new key-value pair into the hashtable. If the key has already been in the
    /// table, the value will be overridden.
    fn insert(&self, key: K, value: V, _outer_map: &AtomicPtr<Self>, guard: &'guard Guard) {
        let mut new_slot = SharedPtr::from_box(Box::new(KVPair { key, value }));
        let (_, new_entry, _) = Self::unwrap_slot(new_slot);
        // new_entry is just created from `key`, so the unwrap() is safe here.
        #[allow(clippy::option_unwrap_used)]
        let new_key = &new_entry.unwrap().key;
        let slot_idx0 = self.get_index(0, new_key);
        let slot_idx1 = self.get_index(1, new_key);
        loop {
            let (slot_idx, slot0, slot1) = self.find(new_key, slot_idx0, slot_idx1, guard);
            let (slot_idx, target_slot, is_replcace) = match slot_idx {
                Some(tbl_idx) => {
                    // The key has already been in the table, we need to replace the value.
                    if tbl_idx == 0 {
                        (Some(&slot_idx0), slot0, true)
                    } else {
                        (Some(&slot_idx1), slot1, true)
                    }
                }
                None => {
                    // The key is a new one, check if we have an empty slot.
                    if Self::slot_is_empty(slot0) {
                        (Some(&slot_idx0), slot0, false)
                    } else if Self::slot_is_empty(slot1) {
                        (Some(&slot_idx1), slot1, false)
                    } else {
                        // Both slots are occupied, we need a relocation.
                        (None, slot0, false)
                    }
                }
            };

            if let Some(slot_idx) = slot_idx {
                // We found the key exists or we have an empty slot,
                // just replace the slot with the new one.

                // update the relocation count.
                new_slot = Self::set_rlcount(new_slot, Self::get_rlcount(target_slot), guard);

                match self.tables[slot_idx.tbl_idx][slot_idx.slot_idx].compare_and_set(
                    target_slot,
                    new_slot,
                    Ordering::SeqCst,
                    guard,
                ) {
                    Ok(old_slot) => {
                        if !is_replcace {
                            self.size.fetch_add(1, Ordering::SeqCst);
                        }
                        Self::defer_drop_ifneed(old_slot, guard);
                        return;
                    }
                    Err(err) => {
                        new_slot = err.1; // the snapshot is not valid, try again.
                        continue;
                    }
                }
            } else {
                // We meet a hash collision here, relocate the first slot.
                if self.relocate(slot_idx0, guard) {
                    continue;
                } else {
                    // The relocation failed! Must resize the table.
                    self.resize();
                }
            }
        }
    }

    /// Remove a key from the map.
    /// TODO: we can return the removed value.
    fn remove(&self, key: &K, _outer_map: &AtomicPtr<Self>, guard: &'guard Guard) -> bool {
        let slot_idx0 = self.get_index(0, key);
        let slot_idx1 = self.get_index(1, key);
        let new_slot = SharedPtr::null();
        loop {
            let (tbl_idx, slot0, slot1) = self.find(key, slot_idx0, slot_idx1, guard);
            let tbl_idx = match tbl_idx {
                Some(idx) => idx,
                None => return false, // The key does not exist.
            };
            if tbl_idx == 0 {
                Self::set_rlcount(new_slot, Self::get_rlcount(slot0), guard);
                match self.tables[0][slot_idx0.slot_idx].compare_and_set(
                    slot0,
                    new_slot,
                    Ordering::SeqCst,
                    guard,
                ) {
                    Ok(old_slot) => {
                        self.size.fetch_sub(1, Ordering::SeqCst);
                        Self::defer_drop_ifneed(old_slot, guard);
                        return true;
                    }
                    Err(_) => continue,
                }
            } else {
                if self.tables[0][slot_idx0.slot_idx]
                    .load(Ordering::SeqCst, guard)
                    .as_raw()
                    != slot0.as_raw()
                {
                    continue;
                }
                Self::set_rlcount(new_slot, Self::get_rlcount(slot1), guard);
                match self.tables[1][slot_idx1.slot_idx].compare_and_set(
                    slot1,
                    new_slot,
                    Ordering::SeqCst,
                    guard,
                ) {
                    Ok(old_slot) => {
                        self.size.fetch_sub(1, Ordering::SeqCst);
                        Self::defer_drop_ifneed(old_slot, guard);
                        return true;
                    }
                    Err(_) => continue,
                }
            }
        }
    }

    /// `find` is similar to `search`, which searches the value corresponding to the key.
    /// The differences are:
    /// 1. `find` will help the relocation if the slot is marked.
    /// 2. `find` will dedup the duplicated keys.
    /// 3. `find` returns three values:
    ///     a> the table index of the slot that has the same key.
    ///     b> the first slot.
    ///     c> the second slot.
    #[allow(clippy::type_complexity)]
    fn find(
        &self,
        key: &K,
        slot_idx0: SlotIndex,
        slot_idx1: SlotIndex,
        guard: &'guard Guard,
    ) -> (
        Option<usize>,
        SharedPtr<'guard, KVPair<K, V>>,
        SharedPtr<'guard, KVPair<K, V>>,
    ) {
        loop {
            let mut result_tbl_index = None;

            // The first round:
            let slot0 = self.get_slot(slot_idx0, guard);
            let (count0_0, entry0, marked0_0) = Self::unwrap_slot(slot0);
            if let Some(pair) = entry0 {
                if let SlotState::Reloc = marked0_0 {
                    self.help_relocate(slot_idx0, false, guard);
                    continue;
                }
                if pair.key.eq(key) {
                    result_tbl_index = Some(0);
                    // We cannot return here, because we may have duplicated keys in both slots.
                    // We must do the deduplication in this method.
                }
            }

            let slot1 = self.get_slot(slot_idx1, guard);
            let (count0_1, entry1, marked0_1) = Self::unwrap_slot(slot1);
            if let Some(pair) = entry1 {
                if let SlotState::Reloc = marked0_1 {
                    self.help_relocate(slot_idx1, false, guard);
                    continue;
                }
                if pair.key.eq(key) {
                    if result_tbl_index.is_some() {
                        // We have a duplicated key in both slots,
                        // need to delete the second one.
                        self.del_dup(slot_idx0, slot0, slot_idx1, slot1, guard);
                    } else {
                        result_tbl_index = Some(1);
                    }
                }
            }

            if result_tbl_index.is_some() {
                return (result_tbl_index, slot0, slot1);
            }

            // The second round:
            let slot0 = self.get_slot(slot_idx0, guard);
            let (count1_0, entry0, marked1_0) = Self::unwrap_slot(slot0);
            if let Some(pair) = entry0 {
                if let SlotState::Reloc = marked1_0 {
                    self.help_relocate(slot_idx0, false, guard);
                    continue;
                }
                if pair.key.eq(key) {
                    result_tbl_index = Some(0);
                }
            }

            let slot1 = self.get_slot(slot_idx1, guard);
            let (count1_1, entry1, marked1_1) = Self::unwrap_slot(slot1);
            if let Some(pair) = entry1 {
                if let SlotState::Reloc = marked1_1 {
                    self.help_relocate(slot_idx1, false, guard);
                    continue;
                }
                if pair.key.eq(key) {
                    if result_tbl_index.is_some() {
                        // We have a duplicated key in both slots,
                        // need to delete the second one.
                        self.del_dup(slot_idx0, slot0, slot_idx1, slot1, guard);
                    } else {
                        result_tbl_index = Some(1);
                    }
                }
            }

            if result_tbl_index.is_some() {
                return (result_tbl_index, slot0, slot1);
            }

            if !Self::check_counter(count0_0, count0_1, count1_0, count1_1) {
                return (None, slot0, slot1);
            }
        }
    }

    /// `help_relocate` helps relocate the slot at `src_idx` to the other corresponding slot.
    #[allow(clippy::option_expect_used)]
    fn help_relocate(&self, src_idx: SlotIndex, initiator: bool, guard: &'guard Guard) {
        loop {
            let mut src_slot = self.get_slot(src_idx, guard);
            while initiator && !Self::slot_is_reloc(src_slot) {
                if Self::slot_is_empty(src_slot) {
                    return;
                }
                let new_slot_with_reloc = src_slot.with_lower_u2(SlotState::Reloc.as_u8());
                // The result will be checked by the `while condition`.
                match self.tables[src_idx.tbl_idx][src_idx.slot_idx].compare_and_set(
                    src_slot,
                    new_slot_with_reloc,
                    Ordering::SeqCst,
                    guard,
                ) {
                    Ok(_) => break,
                    // If the CAS failed, the initiator should try again.
                    Err(current_and_new) => src_slot = current_and_new.0,
                }
            }
            if !Self::slot_is_reloc(src_slot) {
                return;
            }

            let (src_count, src_entry, _) = Self::unwrap_slot(src_slot);
            let dst_idx = self.get_index(
                1 - src_idx.tbl_idx,
                &src_entry.expect("src slot is null").key,
            );
            let dst_slot = self.get_slot(dst_idx, guard);
            let (dst_count, dst_entry, _) = Self::unwrap_slot(dst_slot);

            if dst_entry.is_none() {
                let new_count = if src_count > dst_count {
                    src_count + 1
                } else {
                    dst_count + 1
                };
                if self.get_slot(src_idx, guard).as_raw() != src_slot.as_raw() {
                    continue;
                }
                let new_slot = Self::set_rlcount(src_slot, new_count, guard);

                if self.tables[dst_idx.tbl_idx][dst_idx.slot_idx]
                    .compare_and_set(dst_slot, new_slot, Ordering::SeqCst, guard)
                    .is_ok()
                {
                    let empty_slot = Self::set_rlcount(SharedPtr::null(), src_count + 1, guard);
                    if self.tables[src_idx.tbl_idx][src_idx.slot_idx]
                        .compare_and_set(src_slot, empty_slot, Ordering::SeqCst, guard)
                        .is_ok()
                    {
                        return;
                    }
                }
            }
            // dst is not null
            if src_slot.as_raw() == dst_slot.as_raw() {
                let empty_slot = Self::set_rlcount(SharedPtr::null(), src_count + 1, guard);
                if self.tables[src_idx.tbl_idx][src_idx.slot_idx]
                    .compare_and_set(src_slot, empty_slot, Ordering::SeqCst, guard)
                    .is_ok()
                {
                    // failure cannot happen here.
                }
                return;
            }
            let new_slot_without_mark = Self::set_rlcount(src_slot, src_count + 1, guard)
                .with_lower_u2(SlotState::NullOrKey.as_u8());
            if self.tables[src_idx.tbl_idx][src_idx.slot_idx]
                .compare_and_set(src_slot, new_slot_without_mark, Ordering::SeqCst, guard)
                .is_ok()
            {
                // failure cannot happen here.
            }
            return;
        }
    }

    /// `resize` resizes the table.
    #[allow(clippy::unused_self)]
    fn resize(&self) {
        // FIXME: implement this method.
        unimplemented!("resize() has not been implemented yet.")
    }

    /// `relocate` tries to make the slot in `origin_idx` empty, in order to insert
    /// a new key-value pair into it.
    #[allow(clippy::integer_arithmetic)]
    fn relocate(&self, origin_idx: SlotIndex, guard: &'guard Guard) -> bool {
        let threshold = self.relocation_threshold();
        let mut route = Vec::with_capacity(10); // TODO: optimize this.
        let mut start_level = 0;
        let mut slot_idx = origin_idx;

        // This method consists of two steps:
        // 1. Path Discovery
        //    This step aims to find the cuckoo path which ends with an empty slot,
        //    so we could swap the empty slot backward to the `origin_idx`. Once the
        //    slot at `origin_idx` is empty, the new key-value pair can be inserted.
        // 2. Swap slot
        //    When we have discovered a cuckoo path, we can swap the empty slot backward
        //    to the slot at `origin_idx`.

        'main_loop: loop {
            let mut found = false;
            let mut depth = start_level;
            loop {
                let mut slot = self.get_slot(slot_idx, guard);
                while Self::slot_is_reloc(slot) {
                    self.help_relocate(slot_idx, false, guard);
                    slot = self.get_slot(slot_idx, guard);
                }
                let (_, entry, _) = Self::unwrap_slot(slot);
                if let Some(entry) = entry {
                    let key = &entry.key;

                    // If there are duplicated keys in both slots, we may
                    // meet an endless loop. So we must do the dedup here.
                    let next_slot_idx = self.get_index(1 - slot_idx.tbl_idx, key);
                    let next_slot = self.get_slot(next_slot_idx, guard);
                    let (_, next_entry, _) = Self::unwrap_slot(next_slot);
                    if let Some(pair) = next_entry {
                        if pair.key.eq(key) {
                            if slot_idx.tbl_idx == 0 {
                                self.del_dup(slot_idx, slot, next_slot_idx, next_slot, guard);
                            } else {
                                self.del_dup(next_slot_idx, next_slot, slot_idx, slot, guard);
                            }
                        }
                    }

                    // push the slot into the cuckoo path.
                    if route.len() <= depth {
                        route.push(slot_idx);
                    } else {
                        route[depth] = slot_idx;
                    }
                    slot_idx = next_slot_idx;
                } else {
                    found = true;
                }
                depth += 1;
                if found || depth >= threshold {
                    break;
                }
            }

            if found {
                depth -= 1;
                'slot_swap: for i in (0..depth).rev() {
                    let src_idx = route[i];
                    let mut src_slot = self.get_slot(src_idx, guard);
                    while Self::slot_is_reloc(src_slot) {
                        self.help_relocate(src_idx, false, guard);
                        src_slot = self.get_slot(src_idx, guard);
                    }
                    let (_, entry, _) = Self::unwrap_slot(src_slot);
                    if let Some(pair) = entry {
                        let dst_idx = self.get_index(1 - src_idx.tbl_idx, &pair.key);
                        let (_, dst_entry, _) = self.get_entry(dst_idx, guard);
                        // `dst_entry` should be empty. If it is not, it mains the cuckoo path
                        // has been changed by other threads. Go back to complete the path.
                        if dst_entry.is_some() {
                            start_level = i + 1;
                            slot_idx = dst_idx;
                            continue 'main_loop;
                        }
                        self.help_relocate(src_idx, true, guard);
                    }
                    continue 'slot_swap;
                }
            }
            return found;
        }
    }

    /// `del_dup` deletes the duplicated key. It only deletes the key in the second table.
    fn del_dup(
        &self,
        slot_idx0: SlotIndex,
        slot0: SharedPtr<'guard, KVPair<K, V>>,
        slot_idx1: SlotIndex,
        slot1: SharedPtr<'guard, KVPair<K, V>>,
        guard: &'guard Guard,
    ) {
        if self.get_slot(slot_idx0, guard).as_raw() != slot0.as_raw()
            && self.get_slot(slot_idx1, guard).as_raw() != slot1.as_raw()
        {
            return;
        }
        let (_, entry0, _) = Self::unwrap_slot(slot0);
        let (slot1_count, entry1, _) = Self::unwrap_slot(slot1);
        let mut need_dedup = false;
        if let Some(pair0) = entry0 {
            if let Some(pair1) = entry1 {
                need_dedup = pair0.key.eq(&pair1.key);
            }
        }
        if !need_dedup {
            return;
        }
        let need_free = slot0.as_raw() != slot1.as_raw();
        let empty_slot = Self::set_rlcount(SharedPtr::null(), slot1_count, guard);
        if let Ok(old_slot) = self.tables[slot_idx1.tbl_idx][slot_idx1.slot_idx].compare_and_set(
            slot1,
            empty_slot,
            Ordering::SeqCst,
            guard,
        ) {
            if need_free {
                Self::defer_drop_ifneed(old_slot, guard);
            }
        }
    }

    /// `check_counter` checks the relocation count to decide
    /// whether we need to read the slots again.
    #[allow(clippy::integer_arithmetic)]
    fn check_counter(c00: u8, c01: u8, c10: u8, c11: u8) -> bool {
        // TODO: handle overflow.
        c10 >= c00 + 2 && c11 >= c01 + 2 && c11 >= c00 + 3
    }

    /// `relocation_threshold` returns the threshold of triggering resize.
    fn relocation_threshold(&self) -> usize {
        self.tables[0].len()
    }

    /// `slot_is_reloc` checks if the slot is being relocated.
    fn slot_is_reloc(slot: SharedPtr<'guard, KVPair<K, V>>) -> bool {
        let (_, _, lower_u2) = slot.decompose();
        SlotState::Reloc.as_u8() == lower_u2
    }

    /// `slot_is_empty` checks if the slot is a null pointer.
    fn slot_is_empty(slot: SharedPtr<'guard, KVPair<K, V>>) -> bool {
        let raw = slot.as_raw();
        raw.is_null()
    }

    /// `unwrap_slot` unwraps the slot into three parts:
    /// 1. the relocation count
    /// 2. the key value pair
    /// 3. the state of the slot
    fn unwrap_slot(
        slot: SharedPtr<'guard, KVPair<K, V>>,
    ) -> (u8, Option<&'guard KVPair<K, V>>, SlotState) {
        let (rlcount, raw, lower_u2) = slot.decompose();
        let state = SlotState::from_u8(lower_u2);
        unsafe { (rlcount, raw.as_ref(), state) }
    }

    /// `set_rlcount` sets the relocation count of a slot.
    fn set_rlcount(
        slot: SharedPtr<'guard, KVPair<K, V>>,
        rlcount: u8,
        _: &'guard Guard,
    ) -> SharedPtr<'guard, KVPair<K, V>> {
        slot.with_higher_u8(rlcount)
    }

    /// `get_rlcount` returns the relocation count of a slot.
    fn get_rlcount(slot: SharedPtr<'guard, KVPair<K, V>>) -> u8 {
        let (rlcount, _, _) = slot.decompose();
        rlcount
    }

    /// `get_entry` atomically loads the slot and unwrap it.
    fn get_entry(
        &self,
        slot_idx: SlotIndex,
        guard: &'guard Guard,
    ) -> (u8, Option<&'guard KVPair<K, V>>, SlotState) {
        // TODO: split this method by different memory ordering.
        Self::unwrap_slot(self.get_slot(slot_idx, guard))
    }

    /// `get_slot` atomically loads the slot.
    fn get_slot(
        &self,
        slot_idx: SlotIndex,
        guard: &'guard Guard,
    ) -> SharedPtr<'guard, KVPair<K, V>> {
        self.tables[slot_idx.tbl_idx][slot_idx.slot_idx].load(Ordering::SeqCst, guard)
    }

    /// `get_index` hashes the key and return the slot index.
    #[allow(clippy::result_expect_used, clippy::integer_arithmetic)]
    fn get_index(&self, tbl_idx: usize, key: &K) -> SlotIndex {
        let mut hasher = self.hash_builders[tbl_idx].build_hasher();
        key.hash(&mut hasher);
        let hash_value = usize::try_from(hasher.finish());
        // The conversion from u64 to usize will never fail in a 64-bit env.
        // self.tables[0].len() is always non-zero, so the arithmetic is safe here.
        let slot_idx = hash_value.expect("Cannot convert u64 to usize") % self.tables[0].len();
        SlotIndex { tbl_idx, slot_idx }
    }

    /// `defer_drop_ifneed` tries to defer to drop the slot if not empty.
    #[allow(clippy::as_conversions)]
    fn defer_drop_ifneed(slot: SharedPtr<'guard, KVPair<K, V>>, guard: &'guard Guard) {
        if !Self::slot_is_empty(slot) {
            unsafe {
                // We take over the ownership here.
                // Because only one thread can call this method for the same
                // kv-pair, only one thread can take the ownership. So the
                // as_conversion is safe here.
                guard.defer_destroy(
                    Owned::from_raw(slot.as_raw() as *mut KVPair<K, V>).into_shared(guard),
                );
            }
        }
    }
}

/// `LockFreeCuckooHash` is a lock-free hash table using cuckoo hashing scheme.
/// This implementation is based on the approach discussed in the paper:
///
/// "Nguyen, N., & Tsigas, P. (2014). Lock-Free Cuckoo Hashing. 2014 IEEE 34th International
/// Conference on Distributed Computing Systems, 627-636."
///
/// Cuckoo hashing is an open addressing solution for hash collisions. The basic idea of cuckoo
/// hashing is to resolve collisions by using two or more hash functions instead of only one. In this
/// implementation, we use two hash functions and two arrays (or tables).
///
/// The search operation only looks up two slots, i.e. table[0][hash0(key)] and table[1][hash1(key)].
/// If these two slots do not contain the key, the hash table does not contain the key. So the search operation
/// only takes a constant time in the worst case.
///
/// The insert operation must pay the price for the quick search. The insert operation can only put the key
/// into one of the two slots. However, when both slots are already occupied by other entries, it will be
/// necessary to move other keys to their second locations (or back to their first locations) to make room
/// for the new key, which is called a `relocation`. If the moved key can't be relocated because the other
/// slot of it is also occupied, another `relocation` is required and so on. If relocation is a very long chain
/// or meets a infinite loop, the table should be resized or rehashed.
///
pub struct LockFreeCuckooHash<K, V> {
    /// The inner map will be replaced after resize.
    map: AtomicPtr<MapInner<K, V>>,
}

impl<K, V> std::fmt::Debug for LockFreeCuckooHash<K, V>
where
    K: std::fmt::Debug + Eq + Hash,
    V: std::fmt::Debug,
{
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = pin();
        self.load_inner(&guard).fmt(f)
    }
}

impl<K, V> Default for LockFreeCuckooHash<K, V>
where
    K: Eq + Hash,
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'guard, K, V> LockFreeCuckooHash<K, V>
where
    K: 'guard + Eq + Hash,
{
    /// The default capacity of a new `LockFreeCuckooHash` when created by `LockFreeHashMap::new()`.
    pub const DEFAULT_CAPACITY: usize = 16;

    /// Create an empty `LockFreeCuckooHash` with default capacity.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }

    /// Creates an empty `LockFreeCuckooHash` with the specified capacity.
    #[must_use]
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: AtomicPtr::new(MapInner::with_capacity(
                capacity,
                [RandomState::new(), RandomState::new()],
            )),
        }
    }

    /// Returns the capacity of this hash table.
    #[inline]
    pub fn capacity(&self) -> usize {
        let guard = pin();
        self.load_inner(&guard).capacity()
    }

    /// Returns the number of used slots of this hash table.
    #[inline]
    pub fn size(&self) -> usize {
        let guard = pin();
        self.load_inner(&guard).size()
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// # Example:
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// map.insert(10, 10);
    /// let guard = pin();
    /// let v = map.search_with_guard(&10, &guard);
    /// assert_eq!(*v.unwrap(), 10);
    /// ```
    ///
    #[inline]
    pub fn search_with_guard(&self, key: &K, guard: &'guard Guard) -> Option<&'guard V> {
        self.load_inner(guard).search(key, guard)
    }

    /// Insert a new key-value pair into the hashtable. If the key has already been in the
    /// table, the value will be overridden.
    ///
    /// # Example:
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// map.insert(10, 10);
    /// let guard = pin();
    /// let v1 = map.search_with_guard(&10, &guard);
    /// assert_eq!(*v1.unwrap(), 10);
    /// map.insert(10, 20);
    /// let v2 = map.search_with_guard(&10, &guard);
    /// assert_eq!(*v2.unwrap(), 20);
    /// ```
    ///
    #[inline]
    pub fn insert(&self, key: K, value: V) {
        let guard = pin();
        self.insert_with_guard(key, value, &guard)
    }

    /// Insert a new key-value pair into the hashtable. If the key has already been in the
    /// table, the value will be overridden.
    /// Different from `insert(k, v)`, this method requires a user provided guard.
    ///
    /// # Example:
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// let guard = pin();
    /// map.insert_with_guard(10, 10, &guard);
    /// let v1 = map.search_with_guard(&10, &guard);
    /// assert_eq!(*v1.unwrap(), 10);
    /// map.insert_with_guard(10, 20, &guard);
    /// let v2 = map.search_with_guard(&10, &guard);
    /// assert_eq!(*v2.unwrap(), 20);
    /// ```
    ///
    #[inline]
    pub fn insert_with_guard(&self, key: K, value: V, guard: &'guard Guard) {
        self.load_inner(guard).insert(key, value, &self.map, guard);
    }

    /// Remove a key from the map.
    ///
    /// # Example:
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// let guard = pin();
    /// map.insert(10, 20);
    /// map.remove(&10);
    /// let value = map.search_with_guard(&10, &guard);
    /// assert_eq!(value.is_none(), true);
    /// ```
    ///
    #[inline]
    pub fn remove(&self, key: &K) -> bool {
        let guard = pin();
        self.remove_with_guard(key, &guard)
    }

    /// Remove a key from the map.
    /// Different from `remove(k)`, this method requires a user provided guard.
    ///
    /// # Example:
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// let guard = pin();
    /// map.insert(10, 20);
    /// map.remove_with_guard(&10, &guard);
    /// let value = map.search_with_guard(&10, &guard);
    /// assert_eq!(value.is_none(), true);
    /// ```
    ///
    #[inline]
    pub fn remove_with_guard(&self, key: &K, guard: &'guard Guard) -> bool {
        self.load_inner(guard).remove(key, &self.map, guard)
    }

    /// `load_inner` atomically loads the `MapInner` of hashmap.
    #[allow(clippy::option_unwrap_used)]
    fn load_inner(&self, guard: &'guard Guard) -> &'guard MapInner<K, V> {
        let raw = self.map.load(Ordering::SeqCst, guard).as_raw();
        // map is always not null, so the unsafe code is safe here.
        unsafe { raw.as_ref().unwrap() }
    }
}

#[cfg(test)]
#[allow(clippy::as_conver)]
mod tests {
    use super::{pin, LockFreeCuckooHash};
    use rand::Rng;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Instant;
    #[test]
    fn test_single_thread() {
        let capacity: usize = 100_000;
        let load_factor: f32 = 0.3;
        let remove_factor: f32 = 0.1;
        let size = (capacity as f32 * load_factor) as usize;

        let mut base_map: HashMap<u32, u32> = HashMap::with_capacity(capacity);
        let cuckoo_map: LockFreeCuckooHash<u32, u32> = LockFreeCuckooHash::with_capacity(capacity);

        let mut rng = rand::thread_rng();
        let guard = pin();

        for _ in 0..size {
            let key: u32 = rng.gen();
            let value: u32 = rng.gen();

            base_map.insert(key, value);
            cuckoo_map.insert_with_guard(key, value, &guard);

            let r: u8 = rng.gen();
            let need_remove = (r % 10) < ((remove_factor * 10_f32) as u8);
            if need_remove {
                base_map.remove(&key);
                cuckoo_map.remove_with_guard(&key, &guard);
            }
        }

        assert_eq!(base_map.len(), cuckoo_map.size());

        for (key, value) in base_map {
            let value2 = cuckoo_map.search_with_guard(&key, &guard);
            assert_eq!(value, *value2.unwrap());
        }
    }

    #[test]
    fn test_multi_threads() {
        let capacity: usize = 1_000_000;
        let load_factor: f32 = 0.2;
        let num_thread: usize = 4;

        let size = (capacity as f32 * load_factor) as usize;
        let warmup_size = size / 3;

        let mut warmup_entries: Vec<(u32, u32)> = Vec::with_capacity(warmup_size);

        let mut new_insert_entries: Vec<(u32, u32)> = Vec::with_capacity(size - warmup_size);

        let mut base_map: HashMap<u32, u32> = HashMap::with_capacity(capacity);
        let cuckoo_map: LockFreeCuckooHash<u32, u32> = LockFreeCuckooHash::with_capacity(capacity);

        let mut rng = rand::thread_rng();
        let guard = pin();

        for _ in 0..warmup_size {
            let mut key: u32 = rng.gen();
            while base_map.contains_key(&key) {
                key = rng.gen();
            }
            let value: u32 = rng.gen();
            base_map.insert(key, value);
            cuckoo_map.insert_with_guard(key, value, &guard);
            warmup_entries.push((key, value));
        }

        for _ in 0..(size - warmup_size) {
            let mut key: u32 = rng.gen();
            while base_map.contains_key(&key) {
                key = rng.gen();
            }
            let value: u32 = rng.gen();
            new_insert_entries.push((key, value));
            base_map.insert(key, value);
        }

        let mut handles = Vec::with_capacity(num_thread);
        let insert_count = Arc::new(AtomicUsize::new(0));
        let cuckoo_map = Arc::new(cuckoo_map);
        let warmup_entries = Arc::new(warmup_entries);
        let new_insert_entries = Arc::new(new_insert_entries);
        for _ in 0..num_thread {
            let insert_count = insert_count.clone();
            let cuckoo_map = cuckoo_map.clone();
            let warmup_entries = warmup_entries.clone();
            let new_insert_entries = new_insert_entries.clone();
            let handle = std::thread::spawn(move || {
                let guard = pin();
                let mut entry_idx = insert_count.fetch_add(1, Ordering::SeqCst);
                let mut rng = rand::thread_rng();
                while entry_idx < new_insert_entries.len() {
                    // read 5 pairs ,then insert 1 pair.
                    for _ in 0..5 {
                        let rnd_idx: usize = rng.gen_range(0, warmup_entries.len());
                        let warmup_entry = &warmup_entries[rnd_idx];
                        let res = cuckoo_map.search_with_guard(&warmup_entry.0, &guard);
                        assert_eq!(res.is_some(), true);
                        assert_eq!(*res.unwrap(), warmup_entry.1);
                    }
                    let insert_pair = &new_insert_entries[entry_idx];
                    cuckoo_map.insert_with_guard(insert_pair.0, insert_pair.1, &guard);
                    entry_idx = insert_count.fetch_add(1, Ordering::SeqCst);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        for (k, v) in base_map {
            let v2 = cuckoo_map.search_with_guard(&k, &guard);
            assert_eq!(v, *v2.unwrap());
        }
    }

    #[test]
    #[ignore]
    #[allow(clippy::print_stdout)]
    fn bench_read_write() {
        let num_thread = 4;
        let capacity = 10_000_000;
        let size = 1_000_000;
        let warmup_size = 100_000;
        let num_read_per_write = 19;

        let mut rng = rand::thread_rng();
        let guard = pin();

        let cuckoo_map = LockFreeCuckooHash::with_capacity(capacity);
        let mut warmup_entries = Vec::with_capacity(warmup_size);
        for _ in 0..warmup_size {
            let key: u32 = rng.gen();
            let value: u32 = rng.gen();

            cuckoo_map.insert_with_guard(key, value, &guard);
            warmup_entries.push(key);
        }

        let mut handles = Vec::with_capacity(num_thread);
        let warmup_entries = Arc::new(warmup_entries);
        let cuckoo_map = Arc::new(cuckoo_map);
        let start = Instant::now();
        for _ in 0..num_thread {
            let warmup_entries = warmup_entries.clone();
            let cuckoo_map = cuckoo_map.clone();
            let handle = std::thread::spawn(move || {
                let guard = pin();
                let mut rng = rand::thread_rng();
                for _ in 0..size / num_thread {
                    // 95% read, 5% write
                    for _ in 0..num_read_per_write {
                        let idx: usize = rng.gen_range(0, warmup_entries.len());
                        let key = warmup_entries[idx];
                        cuckoo_map.search_with_guard(&key, &guard);
                    }
                    let key: u32 = rng.gen();
                    let value: u32 = rng.gen();
                    cuckoo_map.insert_with_guard(key, value, &guard);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
        let duration = start.elapsed().as_millis() as f64;
        let throughput = (size * (num_read_per_write + 1)) as f64 / duration;
        let percent_read = num_read_per_write * 100 / (num_read_per_write + 1);
        let percent_write = 100 / (num_read_per_write + 1);
        println!(
            "{}% read + {}% write, total time: {}ms, throughput: {}op/ms",
            percent_read, percent_write, duration, throughput
        );
    }

    #[test]
    #[ignore]
    #[allow(clippy::print_stdout)]
    fn bench_read_only() {
        let num_thread = 4;
        let capacity = 10_000_000;
        let size = 1_000_000;
        let num_read_per_thread = size * 2;
        let mut rng = rand::thread_rng();
        let guard = pin();

        let cuckoo_map = LockFreeCuckooHash::with_capacity(capacity);
        let mut warmup_entries = Vec::with_capacity(size);
        for _ in 0..size {
            let key: u32 = rng.gen();
            let value: u32 = rng.gen();

            cuckoo_map.insert_with_guard(key, value, &guard);
            warmup_entries.push(key);
        }

        let mut handles = Vec::with_capacity(num_thread);
        let warmup_entries = Arc::new(warmup_entries);
        let cuckoo_map = Arc::new(cuckoo_map);
        let start = Instant::now();
        for _ in 0..num_thread {
            let warmup_entries = warmup_entries.clone();
            let cuckoo_map = cuckoo_map.clone();
            let handle = std::thread::spawn(move || {
                let guard = pin();
                let mut rng = rand::thread_rng();
                for _ in 0..num_read_per_thread {
                    let idx: usize = rng.gen_range(0, warmup_entries.len());
                    let key = warmup_entries[idx];
                    cuckoo_map.search_with_guard(&key, &guard);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
        let duration = start.elapsed().as_millis() as f64;
        let throughput = (num_read_per_thread * num_thread) as f64 / duration;
        println!(
            "read only, total time: {}ms, throughput: {}op/ms",
            duration, throughput
        );
    }
}
