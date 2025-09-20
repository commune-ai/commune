#[cfg(test)]
mod tests {
    use super::*;
    use crate as pallet_registry;
    use frame_support::{assert_ok, assert_noop, derive_impl, traits::{ConstU32, Everything}};
    use sp_runtime::BuildStorage;

    type Block = frame_system::mocking::MockBlock<Test>;

    frame_support::construct_runtime!(
        pub enum Test
        {
            System: frame_system,
            Registry: pallet_registry,
        }
    );

    #[derive_impl(frame_system::config_prelude::TestDefaultConfig as frame_system::DefaultConfig)]
    impl frame_system::Config for Test {
        type BaseCallFilter = Everything;
        type Block = Block;
        type RuntimeOrigin = RuntimeOrigin;
        type Nonce = u64;
        type AccountId = u64;
        type Lookup = sp_runtime::traits::IdentityLookup<Self::AccountId>;
        type RuntimeCall = RuntimeCall;
        type RuntimeEvent = RuntimeEvent;
        type BlockHashCount = ConstU32<250>;
        type Version = ();
        type PalletInfo = PalletInfo;
        type AccountData = ();
        type OnNewAccount = ();
        type OnKilledAccount = ();
        type SystemWeightInfo = ();
        type SS58Prefix = ();
        type MaxConsumers = ConstU32<16>;
    }

    impl pallet::Config for Test {
        type RuntimeEvent = RuntimeEvent;
        type MaxKeyLen = ConstU32<64>;
        type MaxValLen = ConstU32<16_384>;
        type WeightInfo = ();
    }

    fn new_test_ext() -> sp_io::TestExternalities {
        let t = frame_system::GenesisConfig::<Test>::default().build_storage().unwrap();
        t.into()
    }

    #[test]
    fn register_works() {
        new_test_ext().execute_with(|| {
            let who = 1u64;
            let key_data = BoundedVec::<u8, <Test as pallet::Config>::MaxKeyLen>::try_from(vec![0u8; 32]).unwrap();
            let value_data = BoundedVec::<u8, <Test as pallet::Config>::MaxValLen>::try_from(b"{\"name\":\"test\"}".to_vec()).unwrap();

            assert_ok!(Registry::register(
                RuntimeOrigin::signed(who),
                KeyType::Sr25519,
                key_data.clone(),
                ValueType::Json,
                value_data.clone()
            ));

            let key = RegistryKey::<Test> {
                key_type: KeyType::Sr25519,
                data: key_data,
            };

            let stored = Registry::registry(&key).unwrap();
            assert_eq!(stored.owner, who);
            assert_eq!(stored.value_type, ValueType::Json);
            assert_eq!(stored.data, value_data);
        });
    }

    #[test]
    fn cannot_register_duplicate() {
        new_test_ext().execute_with(|| {
            let who = 1u64;
            let key_data = BoundedVec::<u8, <Test as pallet::Config>::MaxKeyLen>::try_from(vec![0u8; 32]).unwrap();
            let value_data = BoundedVec::<u8, <Test as pallet::Config>::MaxValLen>::try_from(b"{\"name\":\"test\"}".to_vec()).unwrap();

            assert_ok!(Registry::register(
                RuntimeOrigin::signed(who),
                KeyType::Sr25519,
                key_data.clone(),
                ValueType::Json,
                value_data.clone()
            ));

            assert_noop!(
                Registry::register(
                    RuntimeOrigin::signed(who),
                    KeyType::Sr25519,
                    key_data,
                    ValueType::Json,
                    value_data
                ),
                Error::<Test>::AlreadyExists
            );
        });
    }

    #[test]
    fn update_works() {
        new_test_ext().execute_with(|| {
            let who = 1u64;
            let key_data = BoundedVec::<u8, <Test as pallet::Config>::MaxKeyLen>::try_from(vec![0u8; 32]).unwrap();
            let value_data = BoundedVec::<u8, <Test as pallet::Config>::MaxValLen>::try_from(b"{\"name\":\"test\"}".to_vec()).unwrap();
            let new_value_data = BoundedVec::<u8, <Test as pallet::Config>::MaxValLen>::try_from(b"{\"name\":\"updated\"}".to_vec()).unwrap();

            assert_ok!(Registry::register(
                RuntimeOrigin::signed(who),
                KeyType::Sr25519,
                key_data.clone(),
                ValueType::Json,
                value_data
            ));

            assert_ok!(Registry::update(
                RuntimeOrigin::signed(who),
                KeyType::Sr25519,
                key_data.clone(),
                ValueType::Json,
                new_value_data.clone()
            ));

            let key = RegistryKey::<Test> {
                key_type: KeyType::Sr25519,
                data: key_data,
            };

            let stored = Registry::registry(&key).unwrap();
            assert_eq!(stored.data, new_value_data);
        });
    }

    #[test]
    fn only_owner_can_update() {
        new_test_ext().execute_with(|| {
            let owner = 1u64;
            let other = 2u64;
            let key_data = BoundedVec::<u8, <Test as pallet::Config>::MaxKeyLen>::try_from(vec![0u8; 32]).unwrap();
            let value_data = BoundedVec::<u8, <Test as pallet::Config>::MaxValLen>::try_from(b"{\"name\":\"test\"}".to_vec()).unwrap();

            assert_ok!(Registry::register(
                RuntimeOrigin::signed(owner),
                KeyType::Sr25519,
                key_data.clone(),
                ValueType::Json,
                value_data.clone()
            ));

            assert_noop!(
                Registry::update(
                    RuntimeOrigin::signed(other),
                    KeyType::Sr25519,
                    key_data,
                    ValueType::Json,
                    value_data
                ),
                Error::<Test>::NotOwner
            );
        });
    }

    #[test]
    fn unregister_works() {
        new_test_ext().execute_with(|| {
            let who = 1u64;
            let key_data = BoundedVec::<u8, <Test as pallet::Config>::MaxKeyLen>::try_from(vec![0u8; 32]).unwrap();
            let value_data = BoundedVec::<u8, <Test as pallet::Config>::MaxValLen>::try_from(b"{\"name\":\"test\"}".to_vec()).unwrap();

            assert_ok!(Registry::register(
                RuntimeOrigin::signed(who),
                KeyType::Sr25519,
                key_data.clone(),
                ValueType::Json,
                value_data
            ));

            assert_ok!(Registry::unregister(
                RuntimeOrigin::signed(who),
                KeyType::Sr25519,
                key_data.clone()
            ));

            let key = RegistryKey::<Test> {
                key_type: KeyType::Sr25519,
                data: key_data,
            };

            assert!(Registry::registry(&key).is_none());
        });
    }

    #[test]
    fn invalid_key_length_fails() {
        new_test_ext().execute_with(|| {
            let who = 1u64;
            let bad_key = BoundedVec::<u8, <Test as pallet::Config>::MaxKeyLen>::try_from(vec![0u8; 31]).unwrap(); // Wrong length
            let value_data = BoundedVec::<u8, <Test as pallet::Config>::MaxValLen>::try_from(b"{\"name\":\"test\"}".to_vec()).unwrap();

            assert_noop!(
                Registry::register(
                    RuntimeOrigin::signed(who),
                    KeyType::Sr25519,
                    bad_key,
                    ValueType::Json,
                    value_data
                ),
                Error::<Test>::BadKey
            );
        });
    }
}