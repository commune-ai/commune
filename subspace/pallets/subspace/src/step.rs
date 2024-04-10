use super::*;
use crate::math::*;
use frame_support::{
	storage::{IterableStorageDoubleMap, IterableStorageMap},
};
use substrate_fixed::types::{I110F18, I32F32, I64F64, I96F32};
use sp_std::vec;

impl<T: Config> Pallet<T> {
	pub fn block_step() {
		let block_number: u64 = Self::get_current_block_as_u64();
		RegistrationsPerBlock::<T>::mutate(|val| *val = 0);
		log::debug!("block_step for block: {:?} ", block_number);
		for (netuid, tempo) in <Tempo<T> as IterableStorageMap<u16, u16>>::iter() {

			let new_queued_emission: u64 = Self::calculate_network_emission(netuid);
			PendingEmission::<T>::mutate(netuid, |mut queued| *queued += new_queued_emission);
			log::debug!("netuid_i: {:?} queued_emission: +{:?} ", netuid, new_queued_emission);
			if Self::blocks_until_next_epoch(netuid, tempo, block_number) > 0 {
				continue
			}
			let emission_to_drain: u64 = PendingEmission::<T>::get(netuid).clone();
			Self::epoch(netuid, emission_to_drain);
			PendingEmission::<T>::insert(netuid, 0);
		}
	}


	pub fn epoch(netuid: u16, mut token_emission: u64) {
		// Get subnetwork size.

		let global_params = Self::global_params();
		let subnet_params = Self::subnet_params(netuid);

		let n: u16 = Self::get_subnet_n(netuid);
		let current_block: u64 = Self::get_current_block_as_u64();
		let block_at_registration: Vec<u64> = Self::get_block_at_registration(netuid);

		if n == 0 {
			// 
			return
		}
		
		// FOUNDER DIVIDENDS 
		let founder_key = Self::get_founder(netuid);
		let is_founder_registered = Self::key_registered(netuid, &founder_key);
		let founder_uid = u16::MAX;
		let mut founder_emission: u64 = 0;
		if is_founder_registered {
			let founder_share : u16 = Self::get_founder_share(netuid);
			if founder_share > 0 {
				let founder_emission_ratio: I64F64  = I64F64::from_num(founder_share.min(100))/I64F64::from_num(100);
				founder_emission = (founder_emission_ratio * I64F64::from_num(token_emission)).to_num::<u64>();
				token_emission = token_emission.saturating_sub(founder_emission);
			}
		}

		// ===========
		// == Stake ==
		// ===========

		let mut uid_key_tuples: Vec<(u16, T::AccountId)> = Self::get_uid_key_tuples(netuid);
		let mut stake_64: Vec<I64F64> = vec![I64F64::from_num(0.0); n as usize];
		let mut total_stake_u64: u64 =Self::get_total_subnet_stake(netuid).clone();
		if total_stake_u64 == 0 {
			total_stake_u64 = 1;
		}

		let max_stake = subnet_params.max_stake;

		let stake_u64: Vec<u64> = uid_key_tuples
			.iter()
			.map(|(_, key)| Self::get_stake_for_key(netuid, key).min(max_stake))
			.collect();
		// clip it to the max stake
		let mut stake_f64: Vec<I64F64> = stake_u64.iter().map(|x| I64F64::from_num(x.clone()) /I64F64::from_num(total_stake_u64)).collect();
		let mut stake : Vec<I32F32> = stake_f64.iter().map(|x| I32F32::from_num(x.clone())).collect();
		// Normalize active stake.
		inplace_normalize(&mut stake);

		// =============
		// == Weights (N x N) Sparsified ==
		// =============
		// Access network weights row normalized.
		let last_update_vector: Vec<u64> = Self::get_last_update(netuid);

		let mut weights: Vec<Vec<(u16, u16)>> = vec![vec![]; n as usize];

		let min_weight_stake_f64: I64F64 = I64F64::from_num(global_params.min_weight_stake);


		for (uid_i, mut weights_i) in
			<Weights<T> as IterableStorageDoubleMap<u16, u16, Vec<(u16, u16)>>>::iter_prefix(netuid)
		{
			let mut weight_changed : bool = false;
			// watchout for the overflow

			let weight_age: u64 = current_block.saturating_sub(last_update_vector[uid_i as usize]);
			if weight_age > subnet_params.max_weight_age {
				weight_changed = true;
				weights[uid_i as usize] = vec![];
			} else {
				if (weights_i.len() as usize) < (subnet_params.min_allowed_weights as usize) {
					weight_changed = true;
					weights[uid_i as usize] = vec![];
				}

				

				for (pos, (uid_j, weight_ij)) in weights_i.iter().enumerate() {
					// ignore the weights that are not in the top max allowed weights

					if (pos as u16) <= subnet_params.max_allowed_weights && *uid_j < n {
						// okay , we passed the positioonal check, now check the weight
						let weight_f64 = I64F64::from_num(*weight_ij) / I64F64::from_num(u16::MAX);
						let weight_stake = (stake_f64[uid_i as usize] * weight_f64) * I64F64::from_num(total_stake_u64);
						if weight_stake > min_weight_stake_f64 {
							weights[uid_i as usize].push((*uid_j, *weight_ij));
						} else {
							weight_changed = true;
						}
					} else {
						weight_changed = true;
					}
			}

			}


			if weight_changed {
				// update the weights if it was changed
				<Weights<T>>::insert(netuid, uid_i, weights[uid_i as usize].clone());
			}
		}


		let mut weights : Vec<Vec<(u16, I32F32)>> = weights.iter().map(|x| x.iter().map(|(uid, weight)| (*uid, u16_proportion_to_fixed(*weight))).collect()).collect();
		

		// enabling self voting (if enabled)
		if (!subnet_params.self_vote) {
			weights = mask_diag_sparse(&weights);
		}
		
		// Normalize remaining weights.
		inplace_row_normalize_sparse(&mut weights);

		// =============================
		// ==  Incentive ==
		// =============================

		// convert max u64 to I32F32

		let mut incentive: Vec<I32F32> = vec![I32F32::from_num(0.0); n as usize];
		for (i, sparse_row) in weights.iter().enumerate() {
			// clip based on the max stake
			for (j, value) in sparse_row.iter() {
				incentive[*j as usize] += stake[i] * value;
			}
		}

		// If emission is zero, do an even split.
		if is_zero(&incentive) {
			// no weights set
			for (uid_i, key) in uid_key_tuples.iter() {
				incentive[*uid_i as usize] = I32F32::from_num(1.0);
			}
		}
		inplace_normalize(&mut incentive); // range: I32F32(0, 1)

		// =================================
		// == TRUST ==
		// =================================

		// trust that acts as a multiplier for the incentive
		let trust_ratio: u16 = Self::get_trust_ratio(netuid);
		if trust_ratio > 0 {

			let  trust_share : I32F32 = I32F32::from_num(trust_ratio)/I32F32::from_num(100);
			let incentive_share : I32F32 = I32F32::from_num(1.0).saturating_sub(trust_share);
			let mut trust: Vec<I32F32> = vec![I32F32::from_num(0.0); n as usize];


			for (i, weights_i) in weights.iter().enumerate() {
				for (j, weight_ij) in weights_i.iter() {
					// Compute trust scores: t_j = SUM(i) w_ij * s_i
					// result_j = SUM(i) vector_i * matrix_ij
					if *weight_ij > 0 && 
						stake[i] > I32F32::from_num(subnet_params.min_stake) {
						trust[*j as usize] += I32F32::from_num(1.0);
					}
				}
			}

			inplace_normalize(&mut trust);
			incentive = incentive.iter().zip(trust.iter()).map(|(inc, tru)| (inc * incentive_share) + (tru * trust_share)).collect();
			// save the trust into the trust vector
			Trust::<T>::insert(netuid, trust.iter().map(|xi| fixed_proportion_to_u16(*xi)).collect::<Vec<u16>>());
		}


		// store the incentive
		let cloned_incentive: Vec<u16> = incentive.iter().map(|xi| fixed_proportion_to_u16(*xi)).collect::<Vec<u16>>();
		Incentive::<T>::insert(netuid, cloned_incentive);

		// =================================
		// == Calculate Bonds==
		// =================================

		// Compute bonds delta column normalized.
		let mut bonds: Vec<Vec<(u16, I32F32)>> = weights.clone();
		let mut col_sum: Vec<I32F32> = vec![I32F32::from_num(0.0); n as usize]; // assume square matrix, rows=cols

		for (i, sparse_row) in bonds.iter_mut().enumerate() {
			for (j, value) in sparse_row.iter_mut() {
				*value *= stake[i]; // scale by stake
				col_sum[*j as usize] += *value; // sum the column
			}
		}
		// sum the votes per module
		for sparse_row in bonds.iter_mut() {
			for (j, value) in sparse_row.iter_mut() {
				if col_sum[*j as usize] > I32F32::from_num(0.0 as f32) {
					*value /= col_sum[*j as usize];
				}
			}
		}
		// =================================
		// == Dividends==
		// =================================

		let mut dividends: Vec<I32F32> = vec![I32F32::from_num(0.0); incentive.len()];
		for (i, sparse_row) in bonds.iter().enumerate() {
			for (j, value) in sparse_row.iter() {
				dividends[i] += incentive[*j as usize] * value;
			}
		}

		// If emission is zero, do an even split.
		if is_zero(&dividends) {
			// no weights set
			for (uid_i, key) in uid_key_tuples.iter() {
				dividends[*uid_i as usize] = I32F32::from_num(1.0);
			}
		}
		inplace_normalize(&mut dividends);

		let cloned_dividends: Vec<u16> = dividends.iter().map(|xi| fixed_proportion_to_u16(*xi)).collect::<Vec<u16>>();
		Dividends::<T>::insert(netuid, cloned_dividends);

		// =================================
		// == Emission==
		// =================================
		let mut incentive_ratio: I64F64 =  I64F64::from_num(Self::get_incentive_ratio(netuid) as u64) / I64F64::from_num(100);
		let dividend_ratio: I64F64 = I64F64::from_num(1.0) - incentive_ratio;

		let incentive_emission_float: Vec<I64F64> = incentive
			.clone()
			.iter()
			.map(|x| I64F64::from_num(x.clone()) * I64F64::from_num(token_emission) * incentive_ratio)
			.collect();
		let dividends_emission_float: Vec<I64F64> = dividends
			.clone()
			.iter()
			.map(|x| I64F64::from_num(x.clone()) * I64F64::from_num(token_emission) * dividend_ratio)
			.collect();

		let mut incentive_emission: Vec<u64> =
			incentive_emission_float.iter().map(|e: &I64F64| e.to_num::<u64>()).collect();
		let dividends_emission: Vec<u64> =
			dividends_emission_float.iter().map(|e: &I64F64| e.to_num::<u64>()).collect();

		let mut burn_amount_per_epoch: u64 = Self::get_burn_per_epoch(netuid);

		if is_founder_registered {
			let founder_uid = Self::get_uid_for_key(netuid, &founder_key);
			incentive_emission[founder_uid as usize] = incentive_emission[founder_uid as usize].saturating_add(founder_emission);
		}
			// burn the amount


		let mut emission : Vec<u64> = vec![0; n as usize];
			
		// Emission tuples ( uid_key_tuples, u64 emission)
		let mut founder_share_added: bool = false; // avoid double counting the founder share
		for (module_uid, module_key) in uid_key_tuples.iter() {

			let mut owner_emission_incentive: u64 = incentive_emission[*module_uid as usize];
			let mut owner_dividends_emission: u64 = dividends_emission[*module_uid as usize];
			let mut owner_emission: u64 = owner_emission_incentive + owner_dividends_emission;
			
			// if the owner emission is less than the burn amount
			if burn_amount_per_epoch > owner_emission {
				let burn_into_stake: u64 = burn_amount_per_epoch.saturating_sub(owner_emission);
				
				// decrease the stake if there is remainder
				if burn_into_stake > 0 {
					Self::decrease_stake(netuid, module_key, module_key, burn_into_stake);
				}	
				owner_emission_incentive = 0;
				owner_dividends_emission = 0;			
				// skip the rest of the loop
			} else {
				// eat into incentive first and then into the incentive
				if burn_amount_per_epoch > owner_emission_incentive {
					owner_emission_incentive = 0; 
					// correct the burn amount
					let left_burn_amount_per_epoch = burn_amount_per_epoch.saturating_sub(owner_emission_incentive);
					// apply the burn to the incentive
					owner_dividends_emission = owner_dividends_emission.saturating_sub(left_burn_amount_per_epoch);

				} else {
					// apply the burn to the emission only
					owner_emission_incentive = owner_emission_incentive.saturating_sub(burn_amount_per_epoch);
				}

				emission[*module_uid as usize] = owner_emission_incentive + owner_dividends_emission;
				
				
				if owner_dividends_emission > 0 {
					// get the ownership emission for this key

					let ownership_vector: Vec<(T::AccountId, I64F64)> = Self::get_ownership_ratios(netuid, module_key);
		
					let delegation_fee = Self::get_delegation_fee(netuid, module_key);
					
					// add the ownership
					let total_owner_dividends_emission: u64 = owner_dividends_emission.clone();
					for (delegate_key, delegate_ratio) in ownership_vector.iter() {

						if delegate_key == module_key {
							continue
						}

						let mut dividends_from_delegate : u64 = (I64F64::from_num(total_owner_dividends_emission) * delegate_ratio).to_num::<u64>();
						let to_module: u64 = delegation_fee.mul_floor(dividends_from_delegate);
						let to_delegate: u64 = dividends_from_delegate.saturating_sub(to_module);
						Self::increase_stake(netuid, delegate_key, module_key, to_delegate);
						owner_dividends_emission = owner_dividends_emission.saturating_sub(to_delegate);

					}
				}

				let mut owner_emission: u64 = owner_emission_incentive + owner_dividends_emission;
				// add the emisssion and rm the burn amount
				if owner_emission > 0 {
					// generate the profit shares
					let profit_share_emissions: Vec<(T::AccountId, u64)> = Self::get_profit_share_emissions(module_key.clone(), owner_emission);

					// if there are profit shares, then increase the balance of the profit share key
					if profit_share_emissions.len() > 0 {
						// if there are profit shares, then increase the balance of the profit share key
						for (profit_share_key, profit_share_emission) in profit_share_emissions.iter() {
							// increase the balance of the profit share key
							Self::increase_stake(netuid, profit_share_key, module_key, *profit_share_emission);
						}
					} else {
						// increase it to the module key
						Self::increase_stake(netuid, module_key, module_key, owner_emission);
					}
				}

			}
		}

		

		Emission::<T>::insert(netuid, emission.clone());
	}

	pub fn get_block_at_registration(netuid: u16) -> Vec<u64> {
		let n: usize = Self::get_subnet_n(netuid) as usize;
		let mut block_at_registration: Vec<u64> = vec![0; n];
		for module_uid in 0..n {
			if Keys::<T>::contains_key(netuid, module_uid as u16) {
				block_at_registration[module_uid] =
					Self::get_module_registration_block(netuid, module_uid as u16);
			}
		}
		block_at_registration
	}


	pub fn blocks_until_next_epoch(netuid: u16, tempo: u16, block_number: u64) -> u64 {
		if tempo == 0 {
			return 0
		}
		return (block_number + netuid as u64) % (tempo as u64)
	}

	pub fn get_ownership_ratios_for_uid(netuid: u16, uid: u16) -> Vec<(T::AccountId, I64F64)> {
		return Self::get_ownership_ratios(netuid, &Self::get_key_for_uid(netuid, uid))
	}

	pub fn get_ownership_ratios(
		netuid: u16,
		module_key: &T::AccountId,
	) -> Vec<(T::AccountId, I64F64)> {
		let stake_from_vector: Vec<(T::AccountId, u64)> =
			Self::get_stake_from_vector(netuid, module_key);
		let uid = Self::get_uid_for_key(netuid, module_key);
		let mut total_stake_from: I64F64 = I64F64::from_num(0);

		let mut ownership_vector: Vec<(T::AccountId, I64F64)> = Vec::new();

		for (k, v) in stake_from_vector.clone().into_iter() {
			let ownership = I64F64::from_num(v);
			ownership_vector.push((k.clone(), ownership));
			total_stake_from += ownership;
		}

		// add the module itself, if it has stake of its own
		if total_stake_from == I64F64::from_num(0) {
			ownership_vector.push((module_key.clone(), I64F64::from_num(0)));
		} else {
			ownership_vector =
				ownership_vector.into_iter().map(|(k, v)| (k, v / total_stake_from)).collect();
		}

		return ownership_vector
	}

	pub fn get_ownership_ratios_emission(
		netuid: u16,
		module_key: &T::AccountId,
		emission: u64,
	) -> Vec<(T::AccountId, u64)> {
		let ownership_vector: Vec<(T::AccountId, I64F64)> =
			Self::get_ownership_ratios(netuid, module_key);
		let mut emission_vector: Vec<(T::AccountId, u64)> = Vec::new();

		for (k, v) in ownership_vector {
			let emission_for_delegate = (v * I64F64::from_num(emission)).to_num::<u64>();
			emission_vector.push((k, emission_for_delegate));
		}

		return emission_vector
	}


	pub fn get_burn_per_epoch(netuid:u16) -> u64{

		let n = Self::get_subnet_n(netuid);
		let token_emission: u64 = PendingEmission::<T>::get(netuid).clone();
		let burn_rate: u16 = Self::get_burn_rate().min(100);
		let mut burn_amount_per_epoch : u64 = 0;
		// get the float and convert to u64token_emission
		if burn_rate > 0 {
			let burn_rate_float : I64F64 = (I64F64::from_num(burn_rate)/I64F64::from_num(100)) * (I64F64::from_num(token_emission) / I64F64::from_num(n));
			burn_amount_per_epoch = burn_rate_float.to_num::<u64>();
		}
		return burn_amount_per_epoch

	}

	
}
