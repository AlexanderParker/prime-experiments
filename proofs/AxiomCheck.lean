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
import LiteralCap
import Machine17
import TierA
import PolignacCap
import Spectrum
import LiteralCapTable
import Machine19
import MergeLaw
import TwoTeeth
import Machine19Q
import Machine23
import Machine11
import Machine13Q
import Machine17Q
import Ladder
import DepthSum
import Potential
import Potential19
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
#print axioms LiteralCap.no_run_seven
#print axioms LiteralCap.s_eq
#print axioms LiteralCap.literal_chain_le_six
#print axioms LiteralCap.cap_six_classes_sharp
#print axioms Machine17.w18All
#print axioms Machine17.w25All
#print axioms Machine17.exposed17_iff
#print axioms Machine17.gap_le
#print axioms Machine17.pair_sum_le
#print axioms Machine17.alpha1_certificate
#print axioms Machine17.lemma1_at_17
#print axioms TierA.mem_carrier_of_chain
#print axioms TierA.no_chain_of_carrier_empty
#print axioms TierA.no_maximal_flanks
#print axioms TierA.flanks_17_19
#print axioms TierA.flanks_19_23_nonempty
#print axioms TierA.padding_count_le
#print axioms TierA.padding_at_most_one_below_onset
#print axioms TierA.onset_gate
#print axioms TierA.padding_three_not_excluded
#print axioms TierA.no_adjacent_padded_41
#print axioms TierA.equal_padding_forbidden_classes
#print axioms TierA.equal_padding_forbidden_card
#print axioms TierA.padding_shape_dichotomy
#print axioms TierA.no_adjacent_equal_padded
#print axioms PolignacCap.exists_mul_mod_eq
#print axioms PolignacCap.cap_gcd_1
#print axioms PolignacCap.cap_gcd_3
#print axioms PolignacCap.cap_gcd_15
#print axioms PolignacCap.cap_gcd_105
#print axioms PolignacCap.capOf_le_twelve
#print axioms PolignacCap.cap_gcd_5
#print axioms PolignacCap.cap_gcd_7
#print axioms PolignacCap.cap_gcd_21
#print axioms PolignacCap.cap_gcd_35
#print axioms Spectrum.merged_eq
#print axioms Spectrum.merged_le_spectrum
#print axioms Spectrum.merged_le_spectrum_succ
#print axioms Spectrum.merged_le_of_shallow
#print axioms Spectrum.windowSum_mono
#print axioms Spectrum.qual_le_of_suppressed
#print axioms Spectrum.merged_le_of_suppressed
#print axioms Spectrum.qualifying_of_word
#print axioms Spectrum.merged_le_qual
#print axioms Spectrum.merged_le_of_qual_flat
#print axioms Spectrum.merged_le_of_qual_flat_all
#print axioms Spectrum.merged_le_of_corrected
#print axioms Spectrum.alphabet_ge_floor
#print axioms Spectrum.padded_ge_floor
#print axioms Spectrum.jointCount_antitone
#print axioms LiteralCapTable.cap_table_maximal
#print axioms LiteralCapTable.cap_table_realized
#print axioms LiteralCapTable.literal_chain_le_capC
#print axioms LiteralCapTable.word_length_lt_capC
#print axioms LiteralCapTable.hasRunL_mono
#print axioms LiteralCapTable.capC_le_six
#print axioms LiteralCapTable.cap_two_classes
#print axioms LiteralCapTable.cap_three_classes
#print axioms LiteralCapTable.cap_four_classes
#print axioms LiteralCapTable.cap_six_classes
#print axioms LiteralCapTable.no_cap_five
#print axioms LiteralCapTable.cap_spectrum_counts
#print axioms LiteralCapTable.tripled_teeth_antipode
#print axioms Machine19.sliceAll
#print axioms Machine19.gap_le
#print axioms Machine19.pair_sum_le
#print axioms Machine19.quad_sum_le
#print axioms Machine19.alpha1_certificate
#print axioms Machine19.lemma1_at_19
#print axioms Machine19.shallow_flatness
#print axioms Machine19.exists_exposed_above
#print axioms Machine19.spectrum_four
#print axioms Machine19.spectrum_four_flat
#print axioms Machine19.D_of_shallow_word
#print axioms MergeLaw.sub_mod_eq
#print axioms MergeLaw.interior_gap_mod
#print axioms MergeLaw.floor_of_mod
#print axioms MergeLaw.newgap_le
#print axioms MergeLaw.newgap_le_max
#print axioms MergeLaw.D_of_qualmax
#print axioms TwoTeeth.next_kill_of_lo
#print axioms TwoTeeth.next_kill_of_hi
#print axioms TwoTeeth.kill_spacing
#print axioms TwoTeeth.kill_spacing_min
#print axioms TwoTeeth.kill_period
#print axioms TwoTeeth.kill_spacing_gear
#print axioms TwoTeeth.kill_spacing_min_gear
#print axioms TwoTeeth.teeth_letters
#print axioms TwoTeeth.spacing_from_lo
#print axioms TwoTeeth.spacing_from_hi
#print axioms TwoTeeth.kills_gap_ge
#print axioms TwoTeeth.fuel_span_cap
#print axioms TwoTeeth.fuel_le
#print axioms Machine19.qsliceAll
#print axioms Machine19.chain_facts
#print axioms Machine19.no_big_run
#print axioms Machine19.spectrum_one
#print axioms Machine19.spectrum_two
#print axioms Machine19.spectrum_three
#print axioms Machine19.spectrum_five
#print axioms Machine19.spectrum_ladder
#print axioms Machine19.qual_bound_all
#print axioms Machine19.qual_five_flat
#print axioms Machine19.D_of_word
#print axioms Machine19.opSeq_surj
#print axioms Machine23.killed23_iff
#print axioms Machine23.merge_alphabet
#print axioms Machine23.g23_le
#print axioms Machine23.D_at_19_23

-- round 22: the (D) ladder
#print axioms Machine11.qasm
#print axioms Machine11.chain_facts
#print axioms Machine11.spectrum_ladder
#print axioms Machine11.qual_bound_all
#print axioms Machine11.opSeq_surj
#print axioms Machine13.qasm
#print axioms Machine13.chain_facts
#print axioms Machine13.spectrum_ladder
#print axioms Machine13.qual_bound_all
#print axioms Machine13.opSeq_surj
#print axioms Machine17.qsliceAll
#print axioms Machine17.chain_facts
#print axioms Machine17.spectrum_ladder
#print axioms Machine17.qual_bound_all
#print axioms Machine17.opSeq_surj
#print axioms MergeLaw.pos_le_add
#print axioms MergeLaw.windowSum_telescope
#print axioms MergeLaw.newgap_le_step
#print axioms Ladder.g13_le
#print axioms Ladder.g17_le
#print axioms Ladder.g19_le_of_17
#print axioms Ladder.D_at_11_13
#print axioms Ladder.D_at_13_17
#print axioms Ladder.D_at_17_19
#print axioms Ladder.D_ladder
#print axioms Ladder.D_at_23_29
#print axioms Ladder.D_at_37_41
#print axioms Ladder.criterion_arith
#print axioms DepthSum.window_depth_unique
#print axioms DepthSum.depth_partition
#print axioms DepthSum.mem_reachSet
#print axioms DepthSum.local_factor_5
#print axioms DepthSum.local_factor_13
#print axioms DepthSum.depth_sum_at_13
#print axioms DepthSum.depth_sum_hl_form
#print axioms Potential.chain_le_potential
#print axioms Potential.D_of_potential
#print axioms Potential.windowSum_succ_left
#print axioms Potential.tail_le_potential
#print axioms Potential.merged_le_of_potential
#print axioms Potential19.h19_C1
#print axioms Potential19.h19_C2
#print axioms Potential19.h19_C3
#print axioms Potential19.D_of_word_potential
