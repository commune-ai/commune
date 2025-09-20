#![cfg_attr(not(feature = "std"), no_std)]

// FRAME pallet: Module Registry keyed by arbitrary public keys (Vec<u8>)
// supporting SR25519 and ECDSA key types. Values are JSON bytes (Vec<u8>)
// with extensible ValueType enum. Includes ownership + CRUD extrinsics.

use frame_support::{
    pallet_prelude::*,
    BoundedVec,
};
use frame_system::pallet_prelude::*;
use sp_core::{ecdsa, sr25519};
use sp_runtime::traits::{Hash, Zero};
use scale_info::TypeInfo;

#[frame_support::pallet]
pub mod pallet {
    use super::*;

    #[pallet::config]
    pub trait Config: frame_system::Config {
        /// The overarching event type.
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;

        /// Max length in bytes for a key (public key bytes). ECDSA compressed = 33, SR25519 = 32.
        /// Leave headroom for other types.
        #[pallet::constant]
        type MaxKeyLen: Get<u32>;

        /// Max length in bytes for the stored value (JSON payload or other variants).
        #[pallet::constant]
        type MaxValLen: Get<u32>;

        /// Weight info (optional placeholder).
        type WeightInfo: WeightInfo;
    }

    // -------------------------------
    // Types
    // -------------------------------

    /// Supported public key types.
    #[derive(Clone, Eq, PartialEq, RuntimeDebug, MaxEncodedLen, Encode, Decode, TypeInfo)]
    pub enum KeyType {
        Sr25519,
        Ecdsa,
    }

    /// Value typing to allow for JSON and future variants.
    #[derive(Clone, Eq, PartialEq, RuntimeDebug, MaxEncodedLen, Encode, Decode, TypeInfo)]
    pub enum ValueType {
        /// UTF-8 JSON bytes (not enforced here; producer should provide valid JSON).
        Json,
        /// Opaque bytes for future use (e.g., CID, URL, etc.).
        Bytes,
    }

    /// Record stored per (KeyType, KeyBytes).
    #[derive(Clone, Eq, PartialEq, RuntimeDebug, Encode, Decode, TypeInfo)]
    pub struct ValueRecord<AccountId> {
        pub owner: AccountId,
        pub value_type: ValueType,
        pub data: BoundedVec<u8, <T as Config>::MaxValLen>,
        /// Block number when (last) updated.
        pub updated_at: u32,
    }

    /// Convenience alias for bounded vectors.
    pub type KeyBytes<T> = BoundedVec<u8, <T as Config>::MaxKeyLen>;
    pub type ValBytes<T> = BoundedVec<u8, <T as Config>::MaxValLen>;

    // -------------------------------
    // Pallet struct & genesis
    // -------------------------------

    #[pallet::pallet]
    pub struct Pallet<T>(_);

    #[pallet::storage]
    #[pallet::getter(fn modules)]
    /// Storage mapping: (KeyType, KeyBytes) -> ValueRecord
    pub type Modules<T: Config> = StorageDoubleMap<
        _,
        Blake2_128Concat, KeyType,
        Blake2_128Concat, KeyBytes<T>,
        ValueRecord<T::AccountId>,
        OptionQuery
    >;

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// A module was registered: (owner, key_type, key_hash)
        Registered { owner: T::AccountId, key_type: KeyType, key_hash: T::Hash },
        /// A module was updated.
        Updated { owner: T::AccountId, key_type: KeyType, key_hash: T::Hash },
        /// A module was unregistered.
        Unregistered { owner: T::AccountId, key_type: KeyType, key_hash: T::Hash },
        /// Signature verification result (emits true/false).
        Verified { key_type: KeyType, key_hash: T::Hash, ok: bool },
    }

    #[pallet::error]
    pub enum Error<T> {
        /// Entry already exists for the given key.
        AlreadyExists,
        /// Entry not found.
        NotFound,
        /// Caller is not the owner of the entry.
        NotOwner,
        /// Provided key length exceeds MaxKeyLen or is invalid for the type.
        BadKey,
        /// Provided value length exceeds MaxValLen.
        BadValue,
    }

    #[pallet::hooks]
    impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {}

    // -------------------------------
    // Weights
    // -------------------------------

    pub trait WeightInfo {
        fn register() -> Weight;
        fn update() -> Weight;
        fn unregister() -> Weight;
        fn verify() -> Weight;
    }

    impl WeightInfo for () {
        fn register() -> Weight { Weight::from_parts(10_000, 0) }
        fn update() -> Weight { Weight::from_parts(8_000, 0) }
        fn unregister() -> Weight { Weight::from_parts(8_000, 0) }
        fn verify() -> Weight { Weight::from_parts(10_000, 0) }
    }

    // -------------------------------
    // Extrinsics
    // -------------------------------

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Register a new (key_type, key_bytes) -> JSON mapping. Fails if it already exists.
        #[pallet::call_index(0)]
        #[pallet::weight(T::WeightInfo::register())]
        pub fn register_module(
            origin: OriginFor<T>,
            key_type: KeyType,
            key: KeyBytes<T>,
            value_type: ValueType,
            value: ValBytes<T>,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;
            // validate lengths implicitly via BoundedVec
            ensure!(Self::is_valid_key(&key_type, &key), Error::<T>::BadKey);

            ensure!(Modules::<T>::get(&key_type, &key).is_none(), Error::<T>::AlreadyExists);

            let now = <frame_system::Pallet<T>>::block_number();
            let rec = ValueRecord::<T::AccountId> {
                owner: who.clone(),
                value_type,
                data: value,
                updated_at: Self::bn_to_u32(now),
            };

            Modules::<T>::insert(&key_type, &key, rec);

            let key_hash = T::Hashing::hash(&Self::compose_key_hash_material(&key_type, &key));
            Self::deposit_event(Event::Registered { owner: who, key_type, key_hash });
            Ok(())
        }

        /// Update an existing mapping. Only the owner can update.
        #[pallet::call_index(1)]
        #[pallet::weight(T::WeightInfo::update())]
        pub fn update_module(
            origin: OriginFor<T>,
            key_type: KeyType,
            key: KeyBytes<T>,
            new_value_type: ValueType,
            new_value: ValBytes<T>,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;

            Modules::<T>::try_mutate(&key_type, &key, |maybe_rec| -> DispatchResult {
                let rec = maybe_rec.as_mut().ok_or(Error::<T>::NotFound)?;
                ensure!(rec.owner == who, Error::<T>::NotOwner);
                rec.value_type = new_value_type;
                rec.data = new_value;
                rec.updated_at = Self::bn_to_u32(<frame_system::Pallet<T>>::block_number());
                Ok(())
            })?;

            let key_hash = T::Hashing::hash(&Self::compose_key_hash_material(&key_type, &key));
            Self::deposit_event(Event::Updated { owner: who, key_type, key_hash });
            Ok(())
        }

        /// Remove an existing mapping. Only the owner can remove.
        #[pallet::call_index(2)]
        #[pallet::weight(T::WeightInfo::unregister())]
        pub fn unregister_module(
            origin: OriginFor<T>,
            key_type: KeyType,
            key: KeyBytes<T>,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;

            let rec = Modules::<T>::get(&key_type, &key).ok_or(Error::<T>::NotFound)?;
            ensure!(rec.owner == who, Error::<T>::NotOwner);

            Modules::<T>::remove(&key_type, &key);
            let key_hash = T::Hashing::hash(&Self::compose_key_hash_material(&key_type, &key));
            Self::deposit_event(Event::Unregistered { owner: who, key_type, key_hash });
            Ok(())
        }

        /// Verify a signature against the given key type + key bytes.
        /// `message` is arbitrary; we hash it with blake2_256 for ECDSA; SR25519 takes message bytes directly.
        #[pallet::call_index(3)]
        #[pallet::weight(T::WeightInfo::verify())]
        pub fn verify_signature(
            origin: OriginFor<T>,
            key_type: KeyType,
            key: KeyBytes<T>,
            message: Vec<u8>,
            signature: Vec<u8>,
        ) -> DispatchResult {
            let _ = ensure_signed(origin)?; // anyone can ask to verify
            ensure!(Self::is_valid_key(&key_type, &key), Error::<T>::BadKey);

            let ok = match key_type {
                KeyType::Sr25519 => {
                    if signature.len() != 64 { false } else {
                        let mut sig_bytes = [0u8; 64];
                        sig_bytes.copy_from_slice(&signature);
                        let sig = sr25519::Signature(sig_bytes);
                        // Expect 32-byte public key
                        if key.len() != 32 { false } else {
                            let mut pk = [0u8; 32];
                            pk.copy_from_slice(&key);
                            let pubkey = sr25519::Public(pk);
                            sp_io::crypto::sr25519_verify(&sig, &message, &pubkey)
                        }
                    }
                }
                KeyType::Ecdsa => {
                    // ECDSA expects 65-byte signature (r,s,v) or 64? In Substrate it's 65.
                    if signature.len() != 65 { false } else {
                        let mut sig_bytes = [0u8; 65];
                        sig_bytes.copy_from_slice(&signature);
                        let sig = ecdsa::Signature(sig_bytes);
                        // Compressed secp256k1 pubkey expected: 33 bytes
                        if key.len() != 33 { false } else {
                            let mut pk = [0u8; 33];
                            pk.copy_from_slice(&key);
                            let pubkey = ecdsa::Public(pk);
                            let hash = sp_io::hashing::blake2_256(&message);
                            sp_io::crypto::ecdsa_verify_prehashed(&sig, &hash, &pubkey)
                        }
                    }
                }
            };

            let key_hash = T::Hashing::hash(&Self::compose_key_hash_material(&key_type, &key));
            Self::deposit_event(Event::Verified { key_type, key_hash, ok });
            Ok(())
        }
    }

    // -------------------------------
    // Helpers
    // -------------------------------
    impl<T: Config> Pallet<T> {
        fn bn_to_u32(n: T::BlockNumber) -> u32 {
            // This is safe for typical runtimes; adapt if your BlockNumber > u32
            TryInto::<u32>::try_into(n).unwrap_or(u32::MAX)
        }

        /// Basic key-length validation per type.
        pub fn is_valid_key(key_type: &KeyType, key: &KeyBytes<T>) -> bool {
            match key_type {
                KeyType::Sr25519 => key.len() == 32,
                KeyType::Ecdsa => key.len() == 33, // compressed secp256k1
            }
        }

        /// Material used to compute a stable hash for events/UX (not storage key).
        fn compose_key_hash_material(key_type: &KeyType, key: &KeyBytes<T>) -> Vec<u8> {
            let mut v = Vec::with_capacity(1 + key.len());
            v.push(match key_type { KeyType::Sr25519 => 0, KeyType::Ecdsa => 1 });
            v.extend_from_slice(key);
            v
        }
    }
}

// -------------------------------
// Mock & Tests (minimal skeleton)
// -------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate as pallet_mod_registry;
    use frame_support::{derive_impl, traits::{ConstU32, Everything}};

    type Block = frame_system::mocking::MockBlock<Test>;

    frame_support::construct_runtime!(
        pub enum Test where
            Block = Block,
            NodeBlock = Block,
            UncheckedExtrinsic = frame_system::mocking::MockUncheckedExtrinsic<Test>,
        {
            System: frame_system,
            ModReg: pallet_mod_registry,
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
        type Version = (); type PalletInfo = PalletInfo;
        type AccountData = (); type OnNewAccount = (); type OnKilledAccount = ();
        type SystemWeightInfo = (); type SS58Prefix = (); type MaxConsumers = ConstU32<16>;
    }

    impl pallet::Config for Test {
        type RuntimeEvent = RuntimeEvent;
        type MaxKeyLen = ConstU32<64>;
        type MaxValLen = ConstU32<16_384>;
        type WeightInfo = (); 
    }

    fn new_test_ext() -> sp_io::TestExternalities {
        frame_system::GenesisConfig::<Test>::default().build_storage().unwrap().into()
    }

    #[test]
    fn register_update_remove_flow_works() {
        new_test_ext().execute_with(|| {
            let who = 1u64;
            let key = BoundedVec::<u8, <Test as pallet::Config>::MaxKeyLen>::try_from(vec![0u8;32]).unwrap();
            let val = BoundedVec::<u8, <Test as pallet::Config>::MaxValLen>::try_from(b"{\"name\":\"demo\"}".to_vec()).unwrap();

            assert!(pallet::Pallet::<Test>::register_module(RuntimeOrigin::signed(who), KeyType::Sr25519, key.clone(), ValueType::Json, val.clone()).is_ok());

            let new_val = BoundedVec::<u8, <Test as pallet::Config>::MaxValLen>::try_from(b"{\"name\":\"updated\"}".to_vec()).unwrap();
            assert!(pallet::Pallet::<Test>::update_module(RuntimeOrigin::signed(who), KeyType::Sr25519, key.clone(), ValueType::Json, new_val).is_ok());

            assert!(pallet::Pallet::<Test>::unregister_module(RuntimeOrigin::signed(who), KeyType::Sr25519, key).is_ok());
        });
    }
}
