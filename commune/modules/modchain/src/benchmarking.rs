//! Benchmarking setup for pallet-registry

#![cfg(feature = "runtime-benchmarks")]

use super::*;
use frame_benchmarking::{benchmarks, whitelisted_caller, account};
use frame_system::RawOrigin;
use sp_std::vec;

#[allow(unused)]
use crate::Pallet as Registry;

benchmarks! {
    register {
        let caller: T::AccountId = whitelisted_caller();
        let key_data = vec![0u8; 32];
        let value_data = b"{\"name\":\"benchmark\"}".to_vec();
        
        let key = BoundedVec::<u8, T::MaxKeyLen>::try_from(key_data).unwrap();
        let value = BoundedVec::<u8, T::MaxValLen>::try_from(value_data).unwrap();
    }: _(RawOrigin::Signed(caller.clone()), KeyType::Sr25519, key, ValueType::Json, value)
    verify {
        let registry_key = RegistryKey::<T> {
            key_type: KeyType::Sr25519,
            data: BoundedVec::<u8, T::MaxKeyLen>::try_from(vec![0u8; 32]).unwrap(),
        };
        assert!(Registry::<T>::get(&registry_key).is_some());
    }

    update {
        let caller: T::AccountId = whitelisted_caller();
        let key_data = vec![1u8; 32];
        let value_data = b"{\"name\":\"initial\"}".to_vec();
        let new_value_data = b"{\"name\":\"updated\"}".to_vec();
        
        let key = BoundedVec::<u8, T::MaxKeyLen>::try_from(key_data.clone()).unwrap();
        let value = BoundedVec::<u8, T::MaxValLen>::try_from(value_data).unwrap();
        let new_value = BoundedVec::<u8, T::MaxValLen>::try_from(new_value_data).unwrap();
        
        // Setup: register first
        Registry::<T>::register(
            RawOrigin::Signed(caller.clone()).into(),
            KeyType::Sr25519,
            key.clone(),
            ValueType::Json,
            value
        ).unwrap();
    }: _(RawOrigin::Signed(caller), KeyType::Sr25519, key, ValueType::Json, new_value)
    verify {
        let registry_key = RegistryKey::<T> {
            key_type: KeyType::Sr25519,
            data: BoundedVec::<u8, T::MaxKeyLen>::try_from(vec![1u8; 32]).unwrap(),
        };
        let entry = Registry::<T>::get(&registry_key).unwrap();
        assert_eq!(entry.data.to_vec(), b"{\"name\":\"updated\"}");
    }

    unregister {
        let caller: T::AccountId = whitelisted_caller();
        let key_data = vec![2u8; 32];
        let value_data = b"{\"name\":\"to_remove\"}".to_vec();
        
        let key = BoundedVec::<u8, T::MaxKeyLen>::try_from(key_data.clone()).unwrap();
        let value = BoundedVec::<u8, T::MaxValLen>::try_from(value_data).unwrap();
        
        // Setup: register first
        Registry::<T>::register(
            RawOrigin::Signed(caller.clone()).into(),
            KeyType::Sr25519,
            key.clone(),
            ValueType::Json,
            value
        ).unwrap();
    }: _(RawOrigin::Signed(caller), KeyType::Sr25519, key)
    verify {
        let registry_key = RegistryKey::<T> {
            key_type: KeyType::Sr25519,
            data: BoundedVec::<u8, T::MaxKeyLen>::try_from(vec![2u8; 32]).unwrap(),
        };
        assert!(Registry::<T>::get(&registry_key).is_none());
    }

    verify_signature {
        let caller: T::AccountId = whitelisted_caller();
        let key_data = vec![3u8; 32];
        let message = b"test message".to_vec();
        let signature = vec![0u8; 64]; // Dummy signature for benchmark
        
        let key = BoundedVec::<u8, T::MaxKeyLen>::try_from(key_data).unwrap();
    }: _(RawOrigin::Signed(caller), KeyType::Sr25519, key, message, signature)
    verify {
        // Verification event should be emitted
    }

    impl_benchmark_test_suite!(Registry, crate::tests::new_test_ext(), crate::tests::Test);
}