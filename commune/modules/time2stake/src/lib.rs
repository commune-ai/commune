 # start of file
#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;

#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;

#[cfg(feature = "runtime-benchmarks")]
mod benchmarking;

pub mod weights;

#[frame_support::pallet]
pub mod pallet {
    use frame_support::{
        pallet_prelude::*,
        traits::{Currency, ExistenceRequirement, ReservableCurrency, WithdrawReasons},
    };
    use frame_system::pallet_prelude::*;
    use sp_runtime::{
        traits::{AtLeast32BitUnsigned, CheckedAdd, CheckedSub, Zero},
        ArithmeticError, Permill,
    };
    use sp_std::prelude::*;

    pub type BalanceOf<T> =
        <<T as Config>::Currency as Currency<<T as frame_system::Config>::AccountId>>::Balance;

    #[pallet::pallet]
    pub struct Pallet<T>(_);

    #[pallet::config]
    pub trait Config: frame_system::Config {
        /// The overarching event type.
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;

        /// The currency mechanism.
        type Currency: ReservableCurrency<Self::AccountId>;

        /// Minimum staking period.
        #[pallet::constant]
        type MinimumStakingPeriod: Get<Self::BlockNumber>;

        /// Maximum staking period.
        #[pallet::constant]
        type MaximumStakingPeriod: Get<Self::BlockNumber>;

        /// Base reward rate.
        #[pallet::constant]
        type RewardRate: Get<Permill>;

        /// Weight information for extrinsics in this pallet.
        type WeightInfo: weights::WeightInfo;
    }

    #[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
    pub struct StakeInfo<AccountId, Balance, BlockNumber> {
        /// The owner of the stake.
        pub owner: AccountId,
        /// The amount staked.
        pub amount: Balance,
        /// When the stake was created.
        pub start_time: BlockNumber,
        /// The duration of the stake.
        pub duration: BlockNumber,
        /// Whether rewards should be auto-compounded.
        pub auto_compound: bool,
        /// Last time rewards were claimed or compounded.
        pub last_reward_time: BlockNumber,
    }

    #[pallet::storage]
    #[pallet::getter(fn stakes)]
    pub type Stakes<T: Config> = StorageMap<
        _,
        Blake2_128Concat,
        T::AccountId,
        StakeInfo<T::AccountId, BalanceOf<T>, T::BlockNumber>,
        OptionQuery,
    >;

    #[pallet::storage]
    #[pallet::getter(fn total_staked)]
    pub type TotalStaked<T: Config> = StorageValue<_, BalanceOf<T>, ValueQuery>;

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// Tokens were staked. [account, amount, duration]
        Staked {
            who: T::AccountId,
            amount: BalanceOf<T>,
            duration: T::BlockNumber,
        },
        /// Tokens were unstaked. [account, amount]
        Unstaked {
            who: T::AccountId,
            amount: BalanceOf<T>,
        },
        /// Rewards were claimed. [account, amount]
        RewardsClaimed {
            who: T::AccountId,
            amount: BalanceOf<T>,
        },
        /// Rewards were compounded. [account, amount]
        RewardsCompounded {
            who: T::AccountId,
            amount: BalanceOf<T>,
        },
    }

    #[pallet::error]
    pub enum Error<T> {
        /// No stake found for the account.
        NoStakeFound,
        /// Staking period is too short.
        StakingPeriodTooShort,
        /// Staking period is too long.
        StakingPeriodTooLong,
        /// Staking lock period has not expired.
        StakingLockNotExpired,
        /// Amount is zero.
        ZeroAmount,
        /// Insufficient balance to stake.
        InsufficientBalance,
        /// Insufficient staked amount to unstake.
        InsufficientStakedAmount,
        /// No rewards available.
        NoRewardsAvailable,
    }

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Stake tokens for a specified duration.
        #[pallet::call_index(0)]
        #[pallet::weight(T::WeightInfo::stake())]
        pub fn stake(
            origin: OriginFor<T>,
            #[pallet::compact] amount: BalanceOf<T>,
            duration: T::BlockNumber,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;
            Self::do_stake(who, amount, duration, false)
        }

        /// Stake tokens with auto-compounding option.
        #[pallet::call_index(1)]
        #[pallet::weight(T::WeightInfo::stake_with_compound())]
        pub fn stake_with_compound(
            origin: OriginFor<T>,
            #[pallet::compact] amount: BalanceOf<T>,
            duration: T::BlockNumber,
            auto_compound: bool,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;
            Self::do_stake(who, amount, duration, auto_compound)
        }

        /// Unstake a specific amount of tokens.
        #[pallet::call_index(2)]
        #[pallet::weight(T::WeightInfo::unstake())]
        pub fn unstake(
            origin: OriginFor<T>,
            #[pallet::compact] amount: BalanceOf<T>,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;
            Self::do_unstake(who, Some(amount))
        }

        /// Unstake all tokens.
        #[pallet::call_index(3)]
        #[pallet::weight(T::WeightInfo::unstake_all())]
        pub fn unstake_all(origin: OriginFor<T>) -> DispatchResult {
            let who = ensure_signed(origin)?;
            Self::do_unstake(who, None)
        }

        /// Claim available rewards.
        #[pallet::call_index(4)]
        #[pallet::weight(T::WeightInfo::claim_rewards())]
        pub fn claim_rewards(origin: OriginFor<T>) -> DispatchResult {
            let who = ensure_signed(origin)?;
            Self::do_claim_rewards(who)
        }

        /// Toggle auto-compounding for existing stake.
        #[pallet::call_index(5)]
        #[pallet::weight(T::WeightInfo::toggle_auto_compound())]
        pub fn toggle_auto_compound(origin: OriginFor<T>) -> DispatchResult {
            let who = ensure_signed(origin)?;
            
            Stakes::<T>::try_mutate(&who, |maybe_stake| -> DispatchResult {
                let stake = maybe_stake.as_mut().ok_or(Error::<T>::NoStakeFound)?;
                
                // Toggle auto-compound setting
                stake.auto_compound = !stake.auto_compound;
                
                // If turning on auto-compound, process any pending rewards
                if stake.auto_compound {
                    let current_block = <frame_system::Pallet<T>>::block_number();
                    let rewards = Self::calculate_rewards(&who, current_block)?;
                    
                    if !rewards.is_zero() {
                        stake.amount = stake.amount.checked_add(&rewards)
                            .ok_or(ArithmeticError::Overflow)?;
                        stake.last_reward_time = current_block;
                        
                        // Update total staked
                        <TotalStaked<T>>::try_mutate(|total| -> DispatchResult {
                            *total = total.checked_add(&rewards)
                                .ok_or(ArithmeticError::Overflow)?;
                            Ok(())
                        })?;
                        
                        Self::deposit_event(Event::RewardsCompounded {
                            who: who.clone(),
                            amount: rewards,
                        });
                    }
                }
                
                Ok(())
            })
        }
    }

    impl<T: Config> Pallet<T> {
        // Internal function to handle staking
        fn do_stake(
            who: T::AccountId,
            amount: BalanceOf<T>,
            duration: T::BlockNumber,
            auto_compound: bool,
        ) -> DispatchResult {
            // Validate inputs
            ensure!(!amount.is_zero(), Error::<T>::ZeroAmount);
            ensure!(
                duration >= T::MinimumStakingPeriod::get(),
                Error::<T>::StakingPeriodTooShort
            );
            ensure!(
                duration <= T::MaximumStakingPeriod::get(),
                Error::<T>::StakingPeriodTooLong
            );

            let current_block = <frame_system::Pallet<T>>::block_number();

            // Reserve the tokens
            T::Currency::reserve(&who, amount)
                .map_err(|_| Error::<T>::InsufficientBalance)?;

            // Update or create stake info
            Stakes::<T>::try_mutate(&who, |maybe_stake| -> DispatchResult {
                if let Some(stake) = maybe_stake {
                    // Process any existing rewards before updating stake
                    if !stake.auto_compound {
                        let rewards = Self::calculate_rewards(&who, current_block)?;
                        if !rewards.is_zero() {
                            T::Currency::deposit_into_existing(&who, rewards)
                                .map_err(|_| Error::<T>::InsufficientBalance)?;
                            
                            Self::deposit_event(Event::RewardsClaimed {
                                who: who.clone(),
                                amount: rewards,
                            });
                        }
                    }

                    // Update existing stake
                    stake.amount = stake.amount.checked_add(&amount)
                        .ok_or(ArithmeticError::Overflow)?;
                    stake.duration = duration;
                    stake.auto_compound = auto_compound;
                    stake.last_reward_time = current_block;
                } else {
                    // Create new stake
                    *maybe_stake = Some(StakeInfo {
                        owner: who.clone(),
                        amount,
                        start_time: current_block,
                        duration,
                        auto_compound,
                        last_reward_time: current_block,
                    });
                }
                Ok(())
            })?;

            // Update total staked
            <TotalStaked<T>>::try_mutate(|total| -> DispatchResult {
                *total = total.checked_add(&amount)
                    .ok_or(ArithmeticError::Overflow)?;
                Ok(())
            })?;

            Self::deposit_event(Event::Staked {
                who,
                amount,
                duration,
            });

            Ok(())
        }

        // Internal function to handle unstaking
        fn do_unstake(who: T::AccountId, amount_to_unstake: Option<BalanceOf<T>>) -> DispatchResult {
            Stakes::<T>::try_mutate_exists(&who, |maybe_stake| -> DispatchResult {
                let stake = maybe_stake.as_mut().ok_or(Error::<T>::NoStakeFound)?;
                let current_block = <frame_system::Pallet<T>>::block_number();
                
                // Check if lock period has expired
                let lock_end = stake.start_time.saturating_add(stake.duration);
                ensure!(
                    current_block >= lock_end,
                    Error::<T>::StakingLockNotExpired
                );

                // Calculate rewards
                let rewards = Self::calculate_rewards(&who, current_block)?;

                // Determine amount to unstake
                let unstake_amount = match amount_to_unstake {
                    Some(amount) => {
                        ensure!(amount <= stake.amount, Error::<T>::InsufficientStakedAmount);
                        amount
                    },
                    None => stake.amount,
                };

                // Update stake or remove if fully unstaked
                if unstake_amount == stake.amount {
                    // Fully unstaking
                    *maybe_stake = None;
                } else {
                    // Partially unstaking
                    stake.amount = stake.amount.checked_sub(&unstake_amount)
                        .ok_or(ArithmeticError::Underflow)?;
                    stake.last_reward_time = current_block;
                }

                // Unreserve tokens
                T::Currency::unreserve(&who, unstake_amount);

                // Update total staked
                <TotalStaked<T>>::try_mutate(|total| -> DispatchResult {
                    *total = total.checked_sub(&unstake_amount)
                        .ok_or(ArithmeticError::Underflow)?;
                    Ok(())
                })?;

                // Pay out rewards
                if !rewards.is_zero() {
                    T::Currency::deposit_into_existing(&who, rewards)
                        .map_err(|_| Error::<T>::InsufficientBalance)?;
                    
                    Self::deposit_event(Event::RewardsClaimed {
                        who: who.clone(),
                        amount: rewards,
                    });
                }

                Self::deposit_event(Event::Unstaked {
                    who,
                    amount: unstake_amount,
                });

                Ok(())
            })
        }

        // Internal function to claim rewards
        fn do_claim_rewards(who: T::AccountId) -> DispatchResult {
            Stakes::<T>::try_mutate(&who, |maybe_stake| -> DispatchResult {
                let stake = maybe_stake.as_mut().ok_or(Error::<T>::NoStakeFound)?;
                let current_block = <frame_system::Pallet<T>>::block_number();
                
                // Calculate rewards
                let rewards = Self::calculate_rewards(&who, current_block)?;
                ensure!(!rewards.is_zero(), Error::<T>::NoRewardsAvailable);

                // Update last reward time
                stake.last_reward_time = current_block;

                // Handle rewards based on auto-compound setting
                if stake.auto_compound {
                    // Compound rewards into stake
                    stake.amount = stake.amount.checked_add(&rewards)
                        .ok_or(ArithmeticError::Overflow)?;
                    
                    // Update total staked
                    <TotalStaked<T>>::try_mutate(|total| -> DispatchResult {
                        *total = total.checked_add(&rewards)
                            .ok_or(ArithmeticError::Overflow)?;
                        Ok(())
                    })?;
                    
                    Self::deposit_event(Event::RewardsCompounded {
                        who,
                        amount: rewards,
                    });
                } else {
                    // Pay out rewards directly
                    T::Currency::deposit_into_existing(&who, rewards)
                        .map_err(|_| Error::<T>::InsufficientBalance)?;
                    
                    Self::deposit_event(Event::RewardsClaimed {
                        who,
                        amount: rewards,
                    });
                }

                Ok(())
            })
        }

        // Calculate rewards for a staker
        fn calculate_rewards(
            who: &T::AccountId,
            current_block: T::BlockNumber,
        ) -> Result<BalanceOf<T>, DispatchError> {
            if let Some(stake) = Stakes::<T>::get(who) {
                // Calculate elapsed time since last reward
                let elapsed_time = current_block.saturating_sub(stake.last_reward_time);
                if elapsed_time.is_zero() {
                    return Ok(Zero::zero());
                }

                // Calculate base reward
                let base_rate = T::RewardRate::get();
                let elapsed_time_in_years = Self::blocks_to_years(elapsed_time);
                
                // Apply time-based bonus based on stake duration
                let duration_in_years = Self::blocks_to_years(stake.duration);
                let duration_bonus = Permill::from_percent(
                    (duration_in_years * 10).min(50) as u32
                );
                
                let effective_rate = base_rate.saturating_add(duration_bonus);
                
                // Calculate reward amount
                let reward_amount = effective_rate
                    .mul_floor(stake.amount)
                    .saturating_mul(elapsed_time_in_years.into());
                
                Ok(reward_amount)
            } else {
                Ok(Zero::zero())
            }
        }

        // Helper to convert block numbers to years (simplified)
        fn blocks_to_years(blocks: T::BlockNumber) -> u32 {
            // Assuming 6 second blocks, 14400 blocks per day, 5,256,000 blocks per year
            // This is a simplified calculation and should be adjusted based on your chain's parameters
            let blocks_u32: u32 = blocks.try_into().unwrap_or(0);
            blocks_u32.saturating_div(5_256_000)
        }
    }
}
