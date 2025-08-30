//! # Module Registry Pallet
//!
//! A Substrate pallet for managing a decentralized module registry with multi-chain support.
//! This pallet provides a simple key-value storage system where:
//! - Keys are public keys in various formats (Ed25519, Ethereum, Solana) stored as `Vec<u8>`
//! - Values are IPFS CIDs pointing to module metadata stored as `Vec<u8>`
//!
//! ## Overview
//!
//! This pallet implements:
//! - Multi-chain public key support via flexible `Vec<u8>` keys
//! - IPFS CID storage for off-chain metadata references
//! - Content-addressable metadata via IPFS
//! - Reduced on-chain storage costs (only CIDs stored)
//! - Chain-agnostic design for easy integration
//!
//! ## Storage Design
//!
//! The core storage is a simple StorageMap:
//! - Key: `Vec<u8>` - Public key in various formats (flexible to support all chains)
//! - Value: `Vec<u8>` - IPFS CID pointing to module metadata
//!
//! ## Functionality
//!
//! - `register_module`: Store module metadata CID on-chain
//! - `get_module`: Retrieve module metadata CID by public key
//! - `remove_module`: Delete module from registry
//! - Key validation for different public key formats
//! - CID validation for IPFS references

#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;

#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;

#[cfg(feature = "runtime-benchmarks")]
mod benchmarking;

pub mod weights;
pub use weights::*;

#[frame_support::pallet]
pub mod pallet {
    use super::*;
    use frame_support::pallet_prelude::*;
    use frame_system::pallet_prelude::*;
    extern crate alloc;
    use alloc::vec::Vec;

    #[pallet::pallet]
    pub struct Pallet<T>(_);

    /// The pallet's configuration trait.
    #[pallet::config]
    pub trait Config: frame_system::Config {
        /// The overarching runtime event type.
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
        /// A type representing the weights required by the dispatchables of this pallet.
        type WeightInfo: WeightInfo;
        /// Maximum length for public keys (in bytes)
        #[pallet::constant]
        type MaxKeyLength: Get<u32>;
        /// Maximum length for IPFS CIDs (in bytes)
        #[pallet::constant]
        type MaxCidLength: Get<u32>;
    }

    /// Storage map for module registry.
    /// Maps public keys (Vec<u8>) to IPFS CIDs (Vec<u8>).
    #[pallet::storage]
    #[pallet::getter(fn modules)]
    pub type Modules<T: Config> = StorageMap<
        _,
        Blake2_128Concat,
        BoundedVec<u8, T::MaxKeyLength>,
        BoundedVec<u8, T::MaxCidLength>,
        OptionQuery,
    >;

    /// Events emitted by this pallet.
    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// A module was successfully registered.
        ModuleRegistered {
            /// The public key used as identifier.
            key: BoundedVec<u8, T::MaxKeyLength>,
            /// The IPFS CID of the module metadata.
            cid: BoundedVec<u8, T::MaxCidLength>,
            /// The account who registered the module.
            who: T::AccountId,
        },
        /// A module was successfully updated.
        ModuleUpdated {
            /// The public key used as identifier.
            key: BoundedVec<u8, T::MaxKeyLength>,
            /// The new IPFS CID of the module metadata.
            cid: BoundedVec<u8, T::MaxCidLength>,
            /// The account who updated the module.
            who: T::AccountId,
        },
        /// A module was successfully removed.
        ModuleRemoved {
            /// The public key used as identifier.
            key: BoundedVec<u8, T::MaxKeyLength>,
            /// The account who removed the module.
            who: T::AccountId,
        },
    }

    /// Errors that can be returned by this pallet.
    #[pallet::error]
    pub enum Error<T> {
        /// The module does not exist in the registry.
        ModuleNotFound,
        /// The public key format is invalid.
        InvalidKeyFormat,
        /// The IPFS CID format is invalid.
        InvalidCidFormat,
        /// The public key is too long.
        KeyTooLong,
        /// The IPFS CID is too long.
        CidTooLong,
        /// The public key is empty.
        EmptyKey,
        /// The IPFS CID is empty.
        EmptyCid,
        /// The module already exists in the registry.
        ModuleAlreadyExists,
    }

    /// Dispatchable functions for the module registry pallet.
    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Register a new module in the registry.
        ///
        /// This function stores an IPFS CID for a given public key.
        /// The public key can be in various formats (Ed25519, Ethereum, Solana).
        /// The CID points to the module metadata stored on IPFS.
        ///
        /// # Arguments
        /// * `origin` - The origin of the call (must be signed)
        /// * `key` - The public key to use as identifier (`Vec<u8>`)
        /// * `cid` - The IPFS CID of the module metadata (`Vec<u8>`)
        ///
        /// # Errors
        /// * `InvalidKeyFormat` - If the public key format is invalid
        /// * `InvalidCidFormat` - If the IPFS CID format is invalid
        /// * `ModuleAlreadyExists` - If a module with this key already exists
        #[pallet::call_index(0)]
        #[pallet::weight(T::WeightInfo::register_module())]
        pub fn register_module(origin: OriginFor<T>, key: Vec<u8>, cid: Vec<u8>) -> DispatchResult {
            let who = ensure_signed(origin)?;

            // Validate inputs
            Self::validate_key(&key)?;
            Self::validate_cid(&cid)?;

            // Convert to bounded vectors
            let bounded_key: BoundedVec<u8, T::MaxKeyLength> =
                key.try_into().map_err(|_| Error::<T>::KeyTooLong)?;
            let bounded_cid: BoundedVec<u8, T::MaxCidLength> =
                cid.try_into().map_err(|_| Error::<T>::CidTooLong)?;

            // Check if module already exists
            ensure!(
                !Modules::<T>::contains_key(&bounded_key),
                Error::<T>::ModuleAlreadyExists
            );

            // Store the module
            Modules::<T>::insert(&bounded_key, &bounded_cid);

            // Emit event
            Self::deposit_event(Event::ModuleRegistered {
                key: bounded_key,
                cid: bounded_cid,
                who,
            });

            Ok(())
        }

        /// Update an existing module in the registry.
        ///
        /// This function updates the IPFS CID for an existing public key.
        ///
        /// # Arguments
        /// * `origin` - The origin of the call (must be signed)
        /// * `key` - The public key identifier (`Vec<u8>`)
        /// * `cid` - The new IPFS CID of the module metadata (`Vec<u8>`)
        ///
        /// # Errors
        /// * `ModuleNotFound` - If no module exists with this key
        /// * `InvalidKeyFormat` - If the public key format is invalid
        /// * `InvalidCidFormat` - If the IPFS CID format is invalid
        #[pallet::call_index(1)]
        #[pallet::weight(T::WeightInfo::update_module())]
        pub fn update_module(origin: OriginFor<T>, key: Vec<u8>, cid: Vec<u8>) -> DispatchResult {
            let who = ensure_signed(origin)?;

            // Validate inputs
            Self::validate_key(&key)?;
            Self::validate_cid(&cid)?;

            // Convert to bounded vectors
            let bounded_key: BoundedVec<u8, T::MaxKeyLength> =
                key.try_into().map_err(|_| Error::<T>::KeyTooLong)?;
            let bounded_cid: BoundedVec<u8, T::MaxCidLength> =
                cid.try_into().map_err(|_| Error::<T>::CidTooLong)?;

            // Check if module exists
            ensure!(
                Modules::<T>::contains_key(&bounded_key),
                Error::<T>::ModuleNotFound
            );

            // Update the module
            Modules::<T>::insert(&bounded_key, &bounded_cid);

            // Emit event
            Self::deposit_event(Event::ModuleUpdated {
                key: bounded_key,
                cid: bounded_cid,
                who,
            });

            Ok(())
        }

        /// Remove a module from the registry.
        ///
        /// This function removes a module entry from the storage.
        ///
        /// # Arguments
        /// * `origin` - The origin of the call (must be signed)
        /// * `key` - The public key identifier (`Vec<u8>`)
        ///
        /// # Errors
        /// * `ModuleNotFound` - If no module exists with this key
        /// * `InvalidKeyFormat` - If the public key format is invalid
        #[pallet::call_index(2)]
        #[pallet::weight(T::WeightInfo::remove_module())]
        pub fn remove_module(origin: OriginFor<T>, key: Vec<u8>) -> DispatchResult {
            let who = ensure_signed(origin)?;

            // Validate input
            Self::validate_key(&key)?;

            // Convert to bounded vector
            let bounded_key: BoundedVec<u8, T::MaxKeyLength> =
                key.try_into().map_err(|_| Error::<T>::KeyTooLong)?;

            // Check if module exists
            ensure!(
                Modules::<T>::contains_key(&bounded_key),
                Error::<T>::ModuleNotFound
            );

            // Remove the module
            Modules::<T>::remove(&bounded_key);

            // Emit event
            Self::deposit_event(Event::ModuleRemoved {
                key: bounded_key,
                who,
            });

            Ok(())
        }
    }

    /// Helper functions for validation and utility operations.
    impl<T: Config> Pallet<T> {
        /// Validate a public key format.
        ///
        /// This function performs basic validation on public keys.
        /// It supports various formats by checking length constraints.
        ///
        /// # Arguments
        /// * `key` - The public key to validate
        ///
        /// # Returns
        /// * `Ok(())` if the key is valid
        /// * `Err(Error)` if the key is invalid
        pub fn validate_key(key: &[u8]) -> Result<(), Error<T>> {
            // Check if key is empty
            ensure!(!key.is_empty(), Error::<T>::EmptyKey);

            // Check length constraints
            ensure!(
                key.len() <= T::MaxKeyLength::get() as usize,
                Error::<T>::KeyTooLong
            );

            // Basic format validation for common key types:
            // - Ed25519: 32 bytes
            // - Ethereum: 20 bytes (address) or 64 bytes (public key)
            // - Solana: 32 bytes
            // - Bitcoin: 20 bytes (P2PKH) or 32 bytes (P2WSH)
            match key.len() {
                20 | 32 | 64 => Ok(()),
                _ => {
                    // Allow other lengths for flexibility, but they should be reasonable
                    ensure!(
                        key.len() >= 16 && key.len() <= 128,
                        Error::<T>::InvalidKeyFormat
                    );
                    Ok(())
                }
            }
        }

        /// Validate an IPFS CID format.
        ///
        /// This function performs basic validation on IPFS CIDs.
        /// It supports both CIDv0 and CIDv1 formats.
        ///
        /// # Arguments
        /// * `cid` - The IPFS CID to validate
        ///
        /// # Returns
        /// * `Ok(())` if the CID is valid
        /// * `Err(Error)` if the CID is invalid
        pub fn validate_cid(cid: &[u8]) -> Result<(), Error<T>> {
            // Check if CID is empty
            ensure!(!cid.is_empty(), Error::<T>::EmptyCid);

            // Check length constraints
            ensure!(
                cid.len() <= T::MaxCidLength::get() as usize,
                Error::<T>::CidTooLong
            );

            // Basic CID format validation:
            // CIDv0: starts with "Qm" and is 46 characters (base58)
            // CIDv1: starts with specific multibase prefixes
            // For simplicity, we'll do basic length and character checks

            // Convert to string for validation
            let cid_str = core::str::from_utf8(cid).map_err(|_| Error::<T>::InvalidCidFormat)?;

            // Check minimum and maximum lengths
            ensure!(
                cid_str.len() >= 32 && cid_str.len() <= 128,
                Error::<T>::InvalidCidFormat
            );

            // Basic character validation (alphanumeric + some special chars)
            let valid_chars = cid_str
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
            ensure!(valid_chars, Error::<T>::InvalidCidFormat);

            Ok(())
        }

        /// Get a module's CID by its public key.
        ///
        /// This is a helper function to retrieve module metadata CID.
        ///
        /// # Arguments
        /// * `key` - The public key identifier
        ///
        /// # Returns
        /// * `Some(cid)` if the module exists
        /// * `None` if the module doesn't exist
        pub fn get_module(key: &[u8]) -> Option<BoundedVec<u8, T::MaxCidLength>> {
            let bounded_key: BoundedVec<u8, T::MaxKeyLength> = key.to_vec().try_into().ok()?;
            Modules::<T>::get(&bounded_key)
        }
    }
}
