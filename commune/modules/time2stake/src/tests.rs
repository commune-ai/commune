 # start of file
use crate::{mock::*, Error};
use frame_support::{assert_noop, assert_ok};

#[test]
fn stake_works() {
    new_test_ext().execute_with(|| {
        // Go to block 1
        System::set_block_number(1);
        
        // Stake 100 tokens for 30 days
        assert_ok!(Time2Stake::stake(RuntimeOrigin::signed(1), 100, 30 * 24 * 60 * 10));
        
        // Check that the stake was created
        let stake = Time2Stake::stakes(1).unwrap();
        assert_eq!(stake.amount, 100);
        assert_eq!(stake.duration, 30 * 24 * 60 * 10);
        assert_eq!(stake.auto_compound, false);
        
        // Check total staked
        assert_eq!(Time2Stake::total_staked(), 100);
        
        // Check that the tokens were reserved
        assert_eq!(Balances::free_balance(1), 1000 - 100);
        assert_eq!(Balances::reserved_balance(1), 100);
    });
}

#[test]
fn stake_with_compound_works() {
    new_test_ext().execute_with(|| {
        // Go to block 1
        System::set_block_number(1);
        
        // Stake 100 tokens for 30 days with auto-compound
        assert_ok!(Time2Stake::stake_with_compound(RuntimeOrigin::signed(1), 100, 30 * 24 * 60 * 10, true));
        
        // Check that the stake was created with auto-compound
        let stake = Time2Stake::stakes(1).unwrap();
        assert_eq!(stake.amount, 100);
        assert_eq!(stake.auto_compound, true);
    });
}

#[test]
fn stake_fails_with_zero_amount() {
    new_test_ext().execute_with(|| {
        // Try to stake 0 tokens
        assert_noop!(
            Time2Stake::stake(RuntimeOrigin::signed(1), 0, 30 * 24 * 60 * 10),
            Error::<Test>::ZeroAmount
        );
    });
}

#[test]
fn stake_fails_with_insufficient_balance() {
    new_test_ext().execute_with(|| {
        // Try to stake more than available
        assert_noop!(
            Time2Stake::stake(RuntimeOrigin::signed(1), 2000, 30 * 24 * 60 * 10),
            Error::<Test>::InsufficientBalance
        );
    });
}

#[test]
fn stake_fails_with_too_short_period() {
    new_test_ext().execute_with(|| {
        // Try to stake for too short a period
        assert_noop!(
            Time2Stake::stake(RuntimeOrigin::signed(1), 100, 1),
            Error::<Test>::StakingPeriodTooShort
        );
    });
}

#[test]
fn stake_fails_with_too_long_period() {
    new_test_ext().execute_with(|| {
        // Try to stake for too long a period
        assert_noop!(
            Time2Stake::stake(RuntimeOrigin::signed(1), 100, 1000 * 24 * 60 * 10),
            Error::<Test>::StakingPeriodTooLong
        );
    });
}

#[test]
fn unstake_works_after_lock_period() {
    new_test_ext().execute_with(|| {
        // Go to block 1
        System::set_block_number(1);
        
        // Stake 100 tokens for 30 days
        assert_ok!(Time2Stake::stake(RuntimeOrigin::signed(1), 100, 30 * 24 * 60 * 10));
        
        // Fast forward past lock period
        System::set_block_number(31 * 24 * 60 * 10);
        
        // Unstake 50 tokens
        assert_ok!(Time2Stake::unstake(RuntimeOrigin::signed(1), 50));
        
        // Check that stake was updated
        let stake = Time2Stake::stakes(1).unwrap();
        assert_eq!(stake.amount, 50);
        
        // Check total staked
        assert_eq!(Time2Stake::total_staked(), 50);
        
        // Check balances
        assert_eq!(Balances::free_balance(1), 1000 - 100 + 50);
        assert_eq!(Balances::reserved_balance(1), 50);
    });
}

#[test]
fn unstake_all_works_after_lock_period() {
    new_test_ext().execute_with(|| {
        // Go to block 1
        System::set_block_number(1);
        
        // Stake 100 tokens for 30 days
        assert_ok!(Time2Stake::stake(RuntimeOrigin::signed(1), 100, 30 * 24 * 60 * 10));
        
        // Fast forward past lock period
        System::set_block_number(31 * 24 * 60 * 10);
        
        // Unstake all tokens
        assert_ok!(Time2Stake::unstake_all(RuntimeOrigin::signed(1)));
        
        // Check that stake was removed
        assert!(Time2Stake::stakes(1).is_none());
        
        // Check total staked
        assert_eq!(Time2Stake::total_staked(), 0);
        
        // Check balances
        assert_eq!(Balances::free_balance(1), 1000);
        assert_eq!(Balances::reserved_balance(1), 0);
    });
}

#[test]
fn unstake_fails_during_lock_period() {
    new_test_ext().execute_with(|| {
        // Go to block 1
        System::set_block_number(1);
        
        // Stake 100 tokens for 30 days
        assert_ok!(Time2Stake::stake(RuntimeOrigin::signed(1), 100, 30 * 24 * 60 * 10));
        
        // Try to unstake during lock period
        assert_noop!(
            Time2Stake::unstake(RuntimeOrigin::signed(1), 50),
            Error::<Test>::StakingLockNotExpired
        );
    });
}

#[test]
fn claim_rewards_works() {
    new_test_ext().execute_with(|| {
        // Go to block 1
        System::set_block_number(1);
        
        // Stake 100 tokens for 30 days
        assert_ok!(Time2Stake::stake(RuntimeOrigin::signed(1), 100, 30 * 24 * 60 * 10));
        
        // Fast forward to generate some rewards
        System::set_block_number(10 * 24 * 60 * 10); // 10 days later
        
        // Claim rewards
        assert_ok!(Time2Stake::claim_rewards(RuntimeOrigin::signed(1)));
        
        // Check that last_reward_time was updated
        let stake = Time2Stake::stakes(1).unwrap();
        assert_eq!(stake.last_reward_time, 10 * 24 * 60 * 10);
    });
}

#[test]
fn toggle_auto_compound_works() {
    new_test_ext().execute_with(|| {
        // Go to block 1
        System::set_block_number(1);
        
        // Stake 100 tokens for 30 days without auto-compound
        assert_ok!(Time2Stake::stake(RuntimeOrigin::signed(1), 100, 30 * 24 * 60 * 10));
        
        // Check initial auto-compound setting
        let stake = Time2Stake::stakes(1).unwrap();
        assert_eq!(stake.auto_compound, false);
        
        // Toggle auto-compound
        assert_ok!(Time2Stake::toggle_auto_compound(RuntimeOrigin::signed(1)));
        
        // Check that auto-compound was toggled
        let stake = Time2Stake::stakes(1).unwrap();
        assert_eq!(stake.auto_compound, true);
    });
}
