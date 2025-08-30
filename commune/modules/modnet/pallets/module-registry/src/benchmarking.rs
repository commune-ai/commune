//! Benchmarking setup for pallet-module-registry

use super::*;

use frame_support::BoundedVec;

#[allow(unused)]
use crate::Pallet as ModuleRegistry;
use frame_benchmarking::v2::*;
use frame_system::RawOrigin;

#[benchmarks]
mod benchmarks {
    use super::*;

    #[benchmark]
    fn register_module() {
        let caller: T::AccountId = whitelisted_caller();
        let key = sp_std::vec![1u8; 32]; // Ed25519 key
        let cid = b"QmTestCID123456789012345678901234".to_vec();

        #[extrinsic_call]
        register_module(RawOrigin::Signed(caller), key, cid);

        // Verify that the module was registered
        let bounded_key: BoundedVec<u8, T::MaxKeyLength> =
            sp_std::vec![1u8; 32].try_into().unwrap();
        assert!(Modules::<T>::contains_key(&bounded_key));
    }

    #[benchmark]
    fn update_module() {
        let caller: T::AccountId = whitelisted_caller();
        let key = sp_std::vec![1u8; 32]; // Ed25519 key
        let cid1 = b"QmTestCID123456789012345678901234".to_vec();
        let cid2 = b"QmNewCID1234567890123456789012345".to_vec();

        // First register a module
        let _ = ModuleRegistry::<T>::register_module(
            RawOrigin::Signed(caller.clone()).into(),
            key.clone(),
            cid1,
        );

        #[extrinsic_call]
        update_module(RawOrigin::Signed(caller), key.clone(), cid2);

        // Verify that the module was updated
        let bounded_key: BoundedVec<u8, T::MaxKeyLength> = key.try_into().unwrap();
        assert!(Modules::<T>::contains_key(&bounded_key));
    }

    #[benchmark]
    fn remove_module() {
        let caller: T::AccountId = whitelisted_caller();
        let key = sp_std::vec![1u8; 32]; // Ed25519 key
        let cid = b"QmTestCID123456789012345678901234".to_vec();

        // First register a module
        let _ = ModuleRegistry::<T>::register_module(
            RawOrigin::Signed(caller.clone()).into(),
            key.clone(),
            cid,
        );

        #[extrinsic_call]
        remove_module(RawOrigin::Signed(caller), key.clone());

        // Verify that the module was removed
        let bounded_key: BoundedVec<u8, T::MaxKeyLength> = key.try_into().unwrap();
        assert!(!Modules::<T>::contains_key(&bounded_key));
    }

    impl_benchmark_test_suite!(
        ModuleRegistry,
        crate::mock::new_test_ext(),
        crate::mock::Test
    );
}
