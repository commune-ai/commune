use crate::{mock::*, Error, Event};
use frame_support::{assert_noop, assert_ok, BoundedVec};
extern crate alloc;
use alloc::vec;

#[test]
fn register_module_works() {
    new_test_ext().execute_with(|| {
        // Go past genesis block so events get deposited
        System::set_block_number(1);

        let key = b"test_ed25519_key_32_bytes_long!!".to_vec();
        let cid = b"QmTestCID123456789012345678901234".to_vec();

        // Register a module
        assert_ok!(ModuleRegistry::register_module(
            RuntimeOrigin::signed(1),
            key.clone(),
            cid.clone()
        ));

        // Check that the module was stored
        let bounded_key: BoundedVec<u8, MaxKeyLength> = key.try_into().unwrap();
        let bounded_cid: BoundedVec<u8, MaxCidLength> = cid.try_into().unwrap();

        assert_eq!(
            ModuleRegistry::modules(&bounded_key),
            Some(bounded_cid.clone())
        );

        // Check that the event was emitted
        System::assert_last_event(
            Event::ModuleRegistered {
                key: bounded_key,
                cid: bounded_cid,
                who: 1,
            }
            .into(),
        );
    });
}

#[test]
fn register_module_fails_with_duplicate_key() {
    new_test_ext().execute_with(|| {
        let key = b"test_ed25519_key_32_bytes_long!!".to_vec();
        let cid = b"QmTestCID123456789012345678901234".to_vec();

        // Register a module
        assert_ok!(ModuleRegistry::register_module(
            RuntimeOrigin::signed(1),
            key.clone(),
            cid.clone()
        ));

        // Try to register the same key again
        assert_noop!(
            ModuleRegistry::register_module(RuntimeOrigin::signed(1), key, cid),
            Error::<Test>::ModuleAlreadyExists
        );
    });
}

#[test]
fn register_module_fails_with_empty_key() {
    new_test_ext().execute_with(|| {
        let key = vec![];
        let cid = b"QmTestCID123456789012345678901234".to_vec();

        assert_noop!(
            ModuleRegistry::register_module(RuntimeOrigin::signed(1), key, cid),
            Error::<Test>::EmptyKey
        );
    });
}

#[test]
fn register_module_fails_with_empty_cid() {
    new_test_ext().execute_with(|| {
        let key = b"test_ed25519_key_32_bytes_long!!".to_vec();
        let cid = vec![];

        assert_noop!(
            ModuleRegistry::register_module(RuntimeOrigin::signed(1), key, cid),
            Error::<Test>::EmptyCid
        );
    });
}

#[test]
fn register_module_fails_with_invalid_key_length() {
    new_test_ext().execute_with(|| {
        let key = b"short".to_vec(); // Too short (< 16 bytes)
        let cid = b"QmTestCID123456789012345678901234".to_vec();

        assert_noop!(
            ModuleRegistry::register_module(RuntimeOrigin::signed(1), key, cid),
            Error::<Test>::InvalidKeyFormat
        );
    });
}

#[test]
fn register_module_fails_with_invalid_cid_length() {
    new_test_ext().execute_with(|| {
        let key = b"test_ed25519_key_32_bytes_long!!".to_vec();
        let cid = b"short".to_vec(); // Too short (< 32 bytes)

        assert_noop!(
            ModuleRegistry::register_module(RuntimeOrigin::signed(1), key, cid),
            Error::<Test>::InvalidCidFormat
        );
    });
}

#[test]
fn update_module_works() {
    new_test_ext().execute_with(|| {
        System::set_block_number(1);

        let key = b"test_ed25519_key_32_bytes_long!!".to_vec();
        let cid1 = b"QmTestCID123456789012345678901234".to_vec();
        let cid2 = b"QmNewCID1234567890123456789012345".to_vec();

        // Register a module
        assert_ok!(ModuleRegistry::register_module(
            RuntimeOrigin::signed(1),
            key.clone(),
            cid1
        ));

        // Update the module
        assert_ok!(ModuleRegistry::update_module(
            RuntimeOrigin::signed(1),
            key.clone(),
            cid2.clone()
        ));

        // Check that the module was updated
        let bounded_key: BoundedVec<u8, MaxKeyLength> = key.try_into().unwrap();
        let bounded_cid2: BoundedVec<u8, MaxCidLength> = cid2.try_into().unwrap();

        assert_eq!(
            ModuleRegistry::modules(&bounded_key),
            Some(bounded_cid2.clone())
        );

        // Check that the event was emitted
        System::assert_last_event(
            Event::ModuleUpdated {
                key: bounded_key,
                cid: bounded_cid2,
                who: 1,
            }
            .into(),
        );
    });
}

#[test]
fn update_module_fails_with_nonexistent_key() {
    new_test_ext().execute_with(|| {
        let key = b"test_ed25519_key_32_bytes_long!!".to_vec();
        let cid = b"QmTestCID123456789012345678901234".to_vec();

        assert_noop!(
            ModuleRegistry::update_module(RuntimeOrigin::signed(1), key, cid),
            Error::<Test>::ModuleNotFound
        );
    });
}

#[test]
fn remove_module_works() {
    new_test_ext().execute_with(|| {
        System::set_block_number(1);

        let key = b"test_ed25519_key_32_bytes_long!!".to_vec();
        let cid = b"QmTestCID123456789012345678901234".to_vec();

        // Register a module
        assert_ok!(ModuleRegistry::register_module(
            RuntimeOrigin::signed(1),
            key.clone(),
            cid
        ));

        // Remove the module
        assert_ok!(ModuleRegistry::remove_module(
            RuntimeOrigin::signed(1),
            key.clone()
        ));

        // Check that the module was removed
        let bounded_key: BoundedVec<u8, MaxKeyLength> = key.try_into().unwrap();
        assert_eq!(ModuleRegistry::modules(&bounded_key), None);

        // Check that the event was emitted
        System::assert_last_event(
            Event::ModuleRemoved {
                key: bounded_key,
                who: 1,
            }
            .into(),
        );
    });
}

#[test]
fn remove_module_fails_with_nonexistent_key() {
    new_test_ext().execute_with(|| {
        let key = b"test_ed25519_key_32_bytes_long!!".to_vec();

        assert_noop!(
            ModuleRegistry::remove_module(RuntimeOrigin::signed(1), key),
            Error::<Test>::ModuleNotFound
        );
    });
}

#[test]
fn get_module_helper_works() {
    new_test_ext().execute_with(|| {
        let key = b"test_ed25519_key_32_bytes_long!!".to_vec();
        let cid = b"QmTestCID123456789012345678901234".to_vec();

        // Initially no module
        assert_eq!(ModuleRegistry::get_module(&key), None);

        // Register a module
        assert_ok!(ModuleRegistry::register_module(
            RuntimeOrigin::signed(1),
            key.clone(),
            cid.clone()
        ));

        // Now module should exist
        let bounded_cid: BoundedVec<u8, MaxCidLength> = cid.try_into().unwrap();
        assert_eq!(ModuleRegistry::get_module(&key), Some(bounded_cid));
    });
}

#[test]
fn validate_different_key_formats() {
    new_test_ext().execute_with(|| {
        let cid = b"QmTestCID123456789012345678901234".to_vec();

        // Test Ed25519 key (32 bytes)
        let ed25519_key = vec![0u8; 32];
        assert_ok!(ModuleRegistry::register_module(
            RuntimeOrigin::signed(1),
            ed25519_key,
            cid.clone()
        ));

        // Test Ethereum address (20 bytes)
        let eth_key = vec![1u8; 20];
        assert_ok!(ModuleRegistry::register_module(
            RuntimeOrigin::signed(1),
            eth_key,
            cid.clone()
        ));

        // Test Ethereum public key (64 bytes)
        let eth_pubkey = vec![2u8; 64];
        assert_ok!(ModuleRegistry::register_module(
            RuntimeOrigin::signed(1),
            eth_pubkey,
            cid
        ));
    });
}
