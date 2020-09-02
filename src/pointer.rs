use crossbeam_epoch::Guard;
use std::convert::TryFrom;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
/// `AtomicPtr` is a pointer which can only be manipulated by
/// atomic operations.
#[derive(Debug)]
pub struct AtomicPtr<T: ?Sized> {
    data: AtomicUsize,
    _marker: PhantomData<*mut T>,
}

unsafe impl<T: ?Sized + Send + Sync> Send for AtomicPtr<T> {}
unsafe impl<T: ?Sized + Send + Sync> Sync for AtomicPtr<T> {}

impl<T> AtomicPtr<T> {
    const fn from_usize(data: usize) -> Self {
        Self {
            data: AtomicUsize::new(data),
            _marker: PhantomData,
        }
    }

    pub fn new(value: T) -> Self {
        let b = Box::new(value);
        let raw_ptr = Box::into_raw(b);
        Self::from_usize(raw_ptr as usize)
    }

    pub const fn null() -> Self {
        Self::from_usize(0)
    }

    pub fn load<'g>(&self, ord: Ordering, _: &'g Guard) -> SharedPtr<'g, T> {
        SharedPtr::from_usize(self.data.load(ord))
    }

    pub fn compare_and_set<'g>(
        &self,
        current: SharedPtr<'_, T>,
        new: SharedPtr<'_, T>,
        ord: Ordering,
        _: &'g Guard,
    ) -> Result<SharedPtr<'g, T>, (SharedPtr<'g, T>, SharedPtr<'g, T>)> {
        let new = new.as_usize();
        // TODO: allow different ordering.
        self.data
            .compare_exchange(current.as_usize(), new, ord, ord)
            .map(|_| SharedPtr::from_usize(new))
            .map_err(|current| (SharedPtr::from_usize(current), SharedPtr::from_usize(new)))
    }
}

/// `SharedPtr` is a pointer which can be shared by multi-threads.
/// `SharedPtr` can only be used with 64bit-wide pointer, and the
/// pointer address must be 8-byte aligned.
pub struct SharedPtr<'g, T: 'g> {
    data: usize,
    _marker: PhantomData<(&'g (), *const T)>,
}

impl<T> Clone for SharedPtr<'_, T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            _marker: PhantomData,
        }
    }
}

impl<T> Copy for SharedPtr<'_, T> {}

#[allow(clippy::trivially_copy_pass_by_ref)]
impl<T> SharedPtr<'_, T> {
    pub const fn from_usize(data: usize) -> Self {
        SharedPtr {
            data,
            _marker: PhantomData,
        }
    }

    pub fn from_box(b: Box<T>) -> Self {
        Self::from_raw(Box::into_raw(b))
    }

    pub fn from_raw(raw: *const T) -> Self {
        Self::from_usize(raw as usize)
    }

    pub const fn null() -> Self {
        Self::from_usize(0)
    }

    pub const fn as_usize(&self) -> usize {
        self.data
    }

    fn decompose_lower_u2(data: usize) -> (usize, u8) {
        let mask: usize = 3;
        // The unwrap is safe here, because we have mask the lower 2 bits.
        (data & !mask, u8::try_from(data & mask).unwrap())
    }

    fn decompose_higher_u8(data: usize) -> (u8, usize) {
        let mask: usize = (1 << 56) - 1;
        let higher_u8 = u8::try_from(data >> 56);
        // The unwrap is safe here, because we have shifted 56 bits.
        (higher_u8.unwrap(), data & mask)
    }

    pub fn decompose(&self) -> (u8, *const T, u8) {
        let data = self.data;
        let (data, lower_u2) = Self::decompose_lower_u2(data);
        let (higher_u8, data) = Self::decompose_higher_u8(data);
        (higher_u8, data as *const T, lower_u2)
    }

    pub fn as_raw(&self) -> *const T {
        let (_, raw, _) = self.decompose();
        raw
    }

    pub const fn with_lower_u2(&self, lower_u8: u8) -> Self {
        let mask: usize = 3;
        Self::from_usize(self.data & !mask | lower_u8 as usize)
    }

    pub const fn with_higher_u8(&self, higher_u8: u8) -> Self {
        let data = self.data;
        let mask: usize = (1 << 56) - 1;
        Self::from_usize((data & mask) | ((higher_u8 as usize) << 56))
    }
}
