import BlockedSlots
import Horizon
import Layer
import Supply
import Census
import Bridge
import Gear
import Placement
import Corridor
import Machine13
import MaxGap
open BlockedSlots
#print axioms BlockedSlots.twins_infinite_iff_survivor_in_window
#print axioms BlockedSlots.survivor_in_window_of_gap_bound
#print axioms BlockedSlots.nextGap_spec
#print axioms BlockedSlots.twin_of_centreSurvivor
#print axioms BlockedSlots.covering_of_not_infinite
#print axioms BlockedSlots.survivor_step
#print axioms BlockedSlots.card_blocked_by_le
#print axioms Horizon.exists_prime_factor_lt
#print axioms Horizon.prime_of_no_prime_factor_lt
#print axioms Horizon.twin_of_no_prime_factor_lt
#print axioms Layer.slot_cap
#print axioms Layer.minFac_lt_or_eq
#print axioms Layer.eq_mul_prime_of_minFac_eq
#print axioms Layer.layer_novelty
#print axioms Supply.minFac_mem_gears
#print axioms Supply.card_composites_eq_sum_roots
#print axioms Supply.card_eq_primes_add_sum_roots
#print axioms Supply.roots_ne
#print axioms Census.census_partition
#print axioms Census.comps_eq
#print axioms Census.primes_add_comps
#print axioms Census.primes_eq
#print axioms Census.census_pinned
#print axioms Census.census_pinned_prefix
#print axioms Census.n0_eq_zero_iff
#print axioms Bridge.card_members
#print axioms Bridge.card_comps_members
#print axioms Bridge.card_primes_members
#print axioms Bridge.sum_roots_eq_census
#print axioms Bridge.sum_roots_pinned
#print axioms Bridge.slot_roots_ne
#print axioms Gear.supply_eq_sum_R
#print axioms Gear.sum_R_eq_census
#print axioms Gear.R_le_card_multiples
#print axioms Gear.R_prefix_le
#print axioms Gear.sq_le_of_minFac_eq
#print axioms Gear.R_eq_zero_of_below_sq
#print axioms Gear.semiprime_of_fiber
#print axioms Gear.R_eq_card_partners
#print axioms Gear.mem_partners
#print axioms Placement.prime_mod_six
#print axioms Placement.sign_law
#print axioms Placement.mem_members_iff_slot
#print axioms Placement.slot_injOn_partners
#print axioms Placement.card_slots_of_line
#print axioms Placement.R_slots_eq
#print axioms Corridor.exists_class_in_run
#print axioms Corridor.both_composite_of_class
#print axioms Corridor.both_composite_in_run
#print axioms Corridor.double_slot_in_run
#print axioms Corridor.prime_adjacent_run_le
#print axioms Corridor.product_slotOf
#print axioms Corridor.twin_product_pin
#print axioms Corridor.exposed_iff_mem
#print axioms Corridor.endpoint_law
#print axioms Corridor.endpoint_law_34
#print axioms Corridor.adjacency_law
#print axioms Corridor.no_chain_of_forbidden
#print axioms Corridor.forbidden_first_examples
#print axioms Corridor.forbidden_pairs_count
#print axioms Corridor.n2_packing
#print axioms Machine13.w11
#print axioms Machine13.w16
#print axioms Machine13.exposed13_iff
#print axioms Machine13.gap_le
#print axioms Machine13.pair_sum_le
#print axioms Machine13.gap11_realized
#print axioms Machine13.pair16_realized
#print axioms Machine13.alpha1_certificate
#print axioms Machine13.lemma1_at_13
#print axioms Machine13.tierA_forbidden
#print axioms Machine13.no_11_11_chain
#print axioms MaxGap.uncovered_span_mod_three
#print axioms MaxGap.F_zero_mod_three
#print axioms MaxGap.M_two_mod_three
#print axioms MaxGap.not_max_of_mod_three
