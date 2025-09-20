#![cfg_attr(not(feature = "std"), no_std)]

// FRAME pallet: Module Registry with unified key structure
// Keys are structs containing type prefix and data
// Values include type information alongside the data

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

        /// Max length in bytes for a key (public key bytes).
        #[pallet::constant]
        type MaxKeyLen: Get<u32>;

        /// Max length in bytes for the stored value.
        #[pallet::constant]
        type MaxValLen: Get<u32>;

        /// Weight info.
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
        /// UTF-8 JSON bytes.
        Json,
        /// Opaque bytes for future use.
        Bytes,
    }

    /// Unified key structure with type prefix and data.
    #[derive(Clone, Eq, PartialEq, RuntimeDebug, Encode, Decode, TypeInfo)]
    pub struct RegistryKey<T: Config> {
        pub key_type: KeyType,
        pub data: BoundedVec<u8, T::MaxKeyLen>,
    }

    /// Value structure with type information.
    #[derive(Clone, Eq, PartialEq, RuntimeDebug, Encode, Decode, TypeInfo)]
    pub struct RegistryValue<T: Config, AccountId> {
        pub owner: AccountId,
        pub value_type: ValueType,
        pub data: BoundedVec<u8, T::MaxValLen>,
        pub updated_at: u32,
    }

    // -------------------------------
    // Pallet struct & storage
    // -------------------------------

    #[pallet::pallet]
    pub struct Pallet<T>(_);

    #[pallet::storage]
    #[pallet::getter(fn registry)]
    /// Single unified storage map: RegistryKey -> RegistryValue
    pub type Registry<T: Config> = StorageMap<
        _,
        Blake2_128Concat,
        RegistryKey<T>,
        RegistryValue<T, T::AccountId>,
        OptionQuery
    >;

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// A module was registered.
        Registered { 
            owner: T::AccountId, 
            key: RegistryKey<T>,
            value_type: ValueType,
        },
        /// A module was updated.
        Updated { 
            owner: T::AccountId, 
            key: RegistryKey<T>,
            value_type: ValueType,
        },
        /// A module was unregistered.
        Unregistered { 
            owner: T::AccountId, 
            key: RegistryKey<T>,
        },
        /// Signature verification result.
        Verified { 
            key: RegistryKey<T>,
            ok: bool,
        },
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
        /// Register a new entry in the registry.
        #[pallet::call_index(0)]
        #[pallet::weight(T::WeightInfo::register())]
        pub fn register(
            origin: OriginFor<T>,
            key_type: KeyType,
            key_data: BoundedVec<u8, T::MaxKeyLen>,
            value_type: ValueType,
            value_data: BoundedVec<u8, T::MaxValLen>,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;
            
            // Validate key
            ensure!(Self::is_valid_key(&key_type, &key_data), Error::<T>::BadKey);
            
            let key = RegistryKey::<T> {
                key_type: key_type.clone(),
                data: key_data,
            };
            
            ensure!(Registry::<T>::get(&key).is_none(), Error::<T>::AlreadyExists);
            
            let now = <frame_system::Pallet<T>>::block_number();
            let value = RegistryValue::<T, T::AccountId> {
                owner: who.clone(),
                value_type: value_type.clone(),
                data: value_data,
                updated_at: Self::bn_to_u32(now),
            };
            
            Registry::<T>::insert(&key, value);
            
            Self::deposit_event(Event::Registered { 
                owner: who, 
                key,
                value_type,
            });
            Ok(())
        }

        /// Update an existing entry. Only the owner can update.
        #[pallet::call_index(1)]
        #[pallet::weight(T::WeightInfo::update())]
        pub fn update(
            origin: OriginFor<T>,
            key_type: KeyType,
            key_data: BoundedVec<u8, T::MaxKeyLen>,
            new_value_type: ValueType,
            new_value_data: BoundedVec<u8, T::MaxValLen>,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;
            
            let key = RegistryKey::<T> {
                key_type: key_type.clone(),
                data: key_data,
            };
            
            Registry::<T>::try_mutate(&key, |maybe_value| -> DispatchResult {
                let value = maybe_value.as_mut().ok_or(Error::<T>::NotFound)?;
                ensure!(value.owner == who, Error::<T>::NotOwner);
                value.value_type = new_value_type.clone();
                value.data = new_value_data;
                value.updated_at = Self::bn_to_u32(<frame_system::Pallet<T>>::block_number());
                Ok(())
            })?;
            
            Self::deposit_event(Event::Updated { 
                owner: who, 
                key,
                value_type: new_value_type,
            });
            Ok(())
        }

        /// Remove an existing entry. Only the owner can remove.
        #[pallet::call_index(2)]
        #[pallet::weight(T::WeightInfo::unregister())]
        pub fn unregister(
            origin: OriginFor<T>,
            key_type: KeyType,
            key_data: BoundedVec<u8, T::MaxKeyLen>,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;
            
            let key = RegistryKey::<T> {
                key_type: key_type.clone(),
                data: key_data,
            };
            
            let value = Registry::<T>::get(&key).ok_or(Error::<T>::NotFound)?;
            ensure!(value.owner == who, Error::<T>::NotOwner);
            
            Registry::<T>::remove(&key);
            
            Self::deposit_event(Event::Unregistered { 
                owner: who, 
                key,
            });
            Ok(())
        }

        /// Verify a signature against the given key.
        #[pallet::call_index(3)]
        #[pallet::weight(T::WeightInfo::verify())]
        pub fn verify_signature(
            origin: OriginFor<T>,
            key_type: KeyType,
            key_data: BoundedVec<u8, T::MaxKeyLen>,
            message: Vec<u8>,
            signature: Vec<u8>,
        ) -> DispatchResult {
            let _ = ensure_signed(origin)?;
            ensure!(Self::is_valid_key(&key_type, &key_data), Error::<T>::BadKey);
            
            let ok = match key_type {
                KeyType::Sr25519 => {
                    if signature.len() != 64 || key_data.len() != 32 { 
                        false 
                    } else {
                        let mut sig_bytes = [0u8; 64];
                        sig_bytes.copy_from_slice(&signature);
                        let sig = sr25519::Signature(sig_bytes);
                        let mut pk = [0u8; 32];
                        pk.copy_from_slice(&key_data);
                        let pubkey = sr25519::Public(pk);
                        sp_io::crypto::sr25519_verify(&sig, &message, &pubkey)
                    }
                }
                KeyType::Ecdsa => {
                    if signature.len() != 65 || key_data.len() != 33 { 
                        false 
                    } else {
                        let mut sig_bytes = [0u8; 65];
                        sig_bytes.copy_from_slice(&signature);
                        let sig = ecdsa::Signature(sig_bytes);
                        let mut pk = [0u8; 33];
                        pk.copy_from_slice(&key_data);
                        let pubkey = ecdsa::Public(pk);
                        let hash = sp_io::hashing::blake2_256(&message);
                        sp_io::crypto::ecdsa_verify_prehashed(&sig, &hash, &pubkey)
                    }
                }
            };
            
            let key = RegistryKey::<T> {
                key_type,
                data: key_data,
            };
            
            Self::deposit_event(Event::Verified { key, ok });
            Ok(())
        }
    }

    // -------------------------------
    // Helpers
    // -------------------------------
    impl<T: Config> Pallet<T> {
        fn bn_to_u32(n: T::BlockNumber) -> u32 {
            TryInto::<u32>::try_into(n).unwrap_or(u32::MAX)
        }

        /// Basic key-length validation per type.
        pub fn is_valid_key(key_type: &KeyType, key: &[u8]) -> bool {
            match key_type {
                KeyType::Sr25519 => key.len() == 32,
                KeyType::Ecdsa => key.len() == 33,
            }
        }
    }
}

// Re-export for convenience
pub use pallet::*;