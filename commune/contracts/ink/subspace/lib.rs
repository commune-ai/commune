#![cfg_attr(not(feature = "std"), no_std)]

use ink_lang as ink;


#[ink::contract]
mod subspace {
    use ink_storage::{
        traits::SpreadAllocate,
        Mapping,

    };
    use ink_prelude::{vec::Vec};
    /// Defines the storage of your contract.
    /// Add new fields to the below struct in order
    /// to add new static storage fields to your contract.



    #[ink(storage)]
    #[derive(SpreadAllocate)]
    pub struct Subspace {
        /// Stores a single `bool` value on the storage.
        initial_supply: u64,
        current_supply: u64,
        total_supply: u128, // the total supply
        mint_per_block: u128, // The number of currency minted per block
        votes_per_block: u128, // The number of votes before resetting the block
        votes: Vec<u16>,
        vote_count: u128, 

        // maping users to their 
        user2vote: Mapping<AccountId, Vec<u8>>,

        // user to their  recieved score
        user2score: Mapping<AccountId, Vec<u8>>,

        // user to last update
        user2lastupdate: Mapping<AccountId, u8>,
        users: Vec<AccountId>,

        // endpoint: Mapping<AccountId, Vec<u8>>,
        user2endpoint: Mapping< AccountId, Vec<u8>>,
        user2update: Mapping<AccountId,u64>,
        balance: Mapping<AccountId,u64>,
        // votes: Mapping<AccountId,Vec<u128>>, // votes of additional peers
        stake: Mapping<AccountId,u64>
        
    }

    impl Subspace {
        /// Constructor that initializes the `bool` value to the given `init_value`.
        #[ink(constructor)]
        pub fn new(total_supply: u128, founders: Vec<AccountId>, founder_initial_mints: Vec<u64>) -> Self {

            ink_lang::utils::initialize_contract(|contract| {
                Self::init_state(contract, total_supply, founders, founder_initial_mints)
            })
        }

        fn init_state(&mut self, total_supply: u128, founders: Vec<AccountId>, founder_initial_mints: Vec<u64>)   {
            let caller = Self::env().caller();
            // self.endpoint =  Default::default();
            // self.last_time_update.insert(&caller, &0) ;
            self.balance.insert(&caller, &0);
            self.stake.insert(&caller, &0);
            self.votes = Vec::new();
            self.user2vote = Default::default();
            self.user2score = Default::default();
            self.user2endpoint =  Default::default();
            self.total_supply =  total_supply;
            self.initial_supply = 0;

            // deal with the founders cut
            for (f_idx ,founder) in founders.iter().enumerate() {
                self.balance.insert(&founder, &founder_initial_mints[f_idx]);
                self.initial_supply += founder_initial_mints[f_idx];
            }

            self.current_supply = self.initial_supply;


            
        }
        
        

        /// Constructor that initializes the `bool` value to `false`.
        ///
        /// Constructors can delegate to other constructors.


        /// A message that can be called on instantiated contracts.
        /// This one flips the value of the stored `bool` from `true`
        /// to `false` and vice versa.


        /// Simply returns the current value of our `bool`.
        #[ink(message)]
        pub fn get(&self) -> u8 {
            self.total_supply.try_into().unwrap()
        }
    }

    /// Unit tests in Rust are normally defined within such a `#[cfg(test)]`
    /// module and test functions are marked with a `#[test]` attribute.
    /// The below code is technically just normal Rust code.
    #[cfg(test)]
    mod tests {
        /// Imports all the definitions from the outer scope so we can use them here.
        use super::*;

        /// Imports `ink_lang` so we can use `#[ink::test]`.
        use ink_lang as ink;

        /// We test a simple use case of our contract.
        #[ink::test]
        fn it_works() {
            let mut subspace = Subspace::new(10);
            assert_eq!(subspace.get(), 10);
        }
    }
}
