#!/usr/bin/env python3
"""
filter_bad_vids.py — drop rows referencing un-decodable training videos.

Identified by a parallel decord scan on 2026-05-04: 14/295 mp4s in
/data/Pupil/Pupil_train/vids/ either have a broken
container (no usable video stream) or mid-stream H.264 corruption that
fails on random-frame seek.  The trainer's skip-and-retry hides this but
DPO pair sampling makes the retry storm 2x louder, so we hard-filter.

Usage:
    python filter_bad_vids.py <in.json> [<in.json> ...]

Writes <stem>.vidclean.json next to each input.
"""
import json, os, sys

BAD = {
    "500a_car_jump_starter_from_lidl_unboxing_and_teardown_clean.mp4",
    "boltr_milwaukee_hole_hawg__long_term_quality_clean.mp4",
    "boltr_milwaukee_pump__box_specs_vs_shop_specs_clean.mp4",
    "electromagnetism_maxwells_laws_clean.mp4",
    "identify_chemicals_with_radio_frequencies_nuclear_quadrupole_resonance_mri_without_magnets_clean.mp4",
    "lec_10_second_law_and_available_energy_i_clean.mp4",
    "lec_12_second_law_and_available_energy_iii_clean.mp4",
    "lec_1_introduction_and_fundamental_concepts_clean.mp4",
    "lecture_02_introduction_contd_clean.mp4",
    "lecture_09_boolean_algebra_contd_clean.mp4",
    "measuring_human_digestive_efficiency_vs_a_flame_clean.mp4",
    "measuring_the_amount_of_lead_pb_consumed_when_drinking_from_lead_crystal_glassware_is_it_safe_clean.mp4",
    "mod_01_lec_01_introduction_and_fundamental_concepts_i_clean.mp4",
    "mod_06_lec_06_fluid_statics_part_iii_clean.mp4",
}


def has_bad(row) -> bool:
    s = json.dumps(row)
    return any(b in s for b in BAD)


def main(paths):
    for src in paths:
        with open(src) as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            print(f"  SKIP (not a list): {src}")
            continue
        keep = [r for r in rows if not has_bad(r)]
        dropped = len(rows) - len(keep)
        if src.endswith(".json"):
            dst = src[:-5] + ".vidclean.json"
        else:
            dst = src + ".vidclean.json"
        with open(dst, "w") as f:
            json.dump(keep, f)
        pct = (100.0 * dropped / len(rows)) if rows else 0.0
        print(f"  {os.path.basename(src):60s} {len(rows):5d} -> {len(keep):5d}"
              f"  ({dropped} dropped, {pct:.1f}%)  -> {os.path.basename(dst)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1:])
