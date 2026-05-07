import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv("/home/Pupil/dataset_curation/generation/MMCTAgent/examples/.env")

from pipelines.generator import BenchmarkGenerator
from config import TRANSCRIPT_DIR, VIDEO_DIR, OUTPUT_BASE_DIR, CAT_SYMBOLS, CAT_SPATIAL, CAT_TRANSCRIPT, CAT_ACTION, CAT_FINE_GRAINED

# ==========================================
# 🎛️ KNOBS (CONFIGURATION)
# ==========================================

# 1. Pipeline Selector (Now a List)
# Add any combination of: 'visual', 'audio', 'priority', 'time'
# PIPELINE_MODES = ['visual']
# PIPELINE_MODES = ['visual', 'priority']
PIPELINE_MODES = ['visual', 'audio', 'priority', 'time']

# 2. Target Videos (Now a List)
# VIDEO_NAMES = ["dynamics_lesson_24_work_and_energy_balance_hard_example_clean"]
VIDEO_NAMES = [
    'sea_urchin_development_clean',
    'self_melting_car_voltage_meter_clean',
    'semiconductors_physics_inside_transistors_and_diodes_clean',
    'sexual_reproduction_in_flowering_plants_class_12__male_reproductive_structures__for_neet_exam_clean',
    'silver_nanoprisms_grown_into_structural_colors_by_high_power_leds_clean',
    'simple_diy_curve_tracer_zener_diode_thermal_drift_clean',
    'solving_electric_circuits_clean',
    'soviet_alarm_clock_restoration_elektronika_615m_clean',
    'soviet_digital_clock_analysis_and_restoration_clean',
    'stability_analysis_state_space_3d_visualization_clean',
    'statics_lesson_10_directional_cosines__3d_vector_components_clean',
    'statics_lesson_11_finding_3d_vectors_when_given_coordinates_clean',
    'statics_lesson_12_how_to_find_resultant_force_in_3d_clean',
    'statics_lesson_1_intro_and_newtons_laws_scalers_and_vectors_clean',
    'statics_lesson_2_vector_language_intro_to_vector_addition_clean',
    'statics_lesson_3_the_triangle_rule_for_adding_vectors_to_find_a_resultant_clean',
    'statics_lesson_4_cartesian_notation__vector_addition_explained_clean',
    'statics_lesson_5_triangle_rule_vs_cartesian_components_clean',
    'statics_lesson_8_intro_to_3d_vectors__blue_triangle_equations_clean',
    'statics_lesson_9_drill_problems_practicing_blue_triangle_problems_spherical_coordinates_clean',
    'tektronix_314_storage_oscilloscope_repair_clean',
    'tektronix_tds_744a_teardown_and_repair_2_clean',
    'tektronix_tds_744a_teardown_and_repair_clean',
    'tensors_explained_intuitively_covariant_contravariant_rank_clean',
    'tesla_bm370_vacuum_tube_oscilloscope_teardown_and_restoration_clean',
    'testing_cheap_dashcams_crashcams_dashboard_cameras__car_onboard_dvrs_clean',
    'the_best_projects_from_10_years_of_applied_science_clean',
    'the_hidden_engineering_of_landfills_clean',
    'the_most_confusing_part_of_the_power_grid_clean',
    'the_time_a_pickup_pulled_the_space_shuttle_clean',
    'thermodynamics_and_the_end_of_the_universe_energy_entropy_and_the_fundamental_laws_of_physics_clean',
    'three_phase_electricity_basics_and_calculations_electrical_engineering_clean',
    'torque_levers_and_the_universal_law_of_rotation_clean',
    'traffic_light_garage_parking_sensor_unboxing_teardown_schematic_clean',
    'transformers_explained_how_transformers_work_clean',
    'transistors_field_effect_and_bipolar_transistors_mosfets_and_bjts_clean',
    'trigonometry_easy_to_understand_3d_animation_clean',
    'underwater_laser_cutting_and_silver_sintering_to_make_ceramic_circuit_boards_clean',
    'utrai_2500a_car_jump_starter_inside_the_protection_box_clean',
    'utrai_2500a_car_jump_starter_unboxing_and_teardown_clean',
    'voltage_current_electricity_magnetism_clean',
    'wave_reflection_and_transmission_clean',
    'waves_light_sound_and_the_nature_of_reality_clean',
    'weak_nuclear_force_and_standard_model_of_particle_physics_clean',
    'what_causes_the_pauli_exclusion_principle_clean',
    'what_is_a_black_start_of_the_power_grid_clean',
    'why_construction_projects_always_go_over_budget_clean',
    'why_locomotives_dont_have_tires_clean',
    'why_rivers_move_clean',
    'work_energy_and_power_lesson_conservation_of_energy_clean',
    'work_energy_and_power_lesson_problem_solving_clean',
    'work_energy_and_power_lesson_the_work_energy_theorem_clean',
    'work_energy_and_power_lesson_work_clean',
    'x_ray_backscatter_with_compressed_sensing_algorithm_clean',
    'x_ray_timelapse_of_fluid_movement_in_plants_stop_motion_animation_sensor_teardownrepair_clean'
]

SKIP_LIST = {
    # visual
    ("visual", "ch3pr38_damped_harmonic_oscillation_clean"),
    ("visual", "introduction_to_the_hyperbola_2_of_2_basic_shape__characteristics_clean"),
    ("visual", "lecture_19__linearization_of_non_linear_circuit_containing_bjt_contd_clean"),
    ("visual", "the_petty_cashbook_part_2_clean"),
    ("visual", "w8_l5_example_2_rib_mirror_clean"),
    # audio
    ("audio", "elec2141_digital_circuit_design_lecture_27_clean"),
    ("audio", "graphing_techniques_qa_3_of_3_solving_an_inequality_visually_clean"),
    ("audio", "lecture_22__linear_models_of_amplifiers_part_a_clean"),
    ("audio", "mod02lec11_kinematic_simulation_of_wheeled_mobile_robots_part_2_clean"),
    # priority
    ("priority", "lecture_44__common_collector_and_common_drain_amplifiers_clean"),
    ("priority", "making_liquid_nitrogen_from_scratch_clean"),
    # time
    ("time", "civilization_1_explaining_humanitys_transition_to_agriculture_clean"),
    ("time", "eth_zürich_aise_windowed_attention_and_scaling_laws_clean"),
    ("time", "geo_strategy2_christian_zionism_and_the_middle_east_conflict_clean"),
    ("time", "how_lawn_mower_blades_cut_grass_at_50000_frames_per_second_smarter_every_day_196_clean"),
    ("time", "introduction_to_the_hyperbola_2_of_2_basic_shape__characteristics_clean"),
    ("time", "lecture_19__linearization_of_non_linear_circuit_containing_bjt_contd_clean"),
    ("time", "rust_removal_experiments_electrolysis_clean"),
    ("time", "w6_l3_example_isometric_drawings_clean"),
}

# 3. Category Nudge (Optional)
# Options: CAT_OCR, CAT_SPATIAL, CAT_SEMANTIC, CAT_ACTION, CAT_SIMULATION, CAT_FINE_GRAINED, or None
CATEGORY_NUDGE = None

# 4. Limits
TOTAL_QUESTIONS = 3
# ==========================================

async def main():
    # Outer Loop: Iterate through all videos
    for video_name in VIDEO_NAMES:
        video_path = VIDEO_DIR / f"{video_name}.mp4"
        
        if not video_path.exists():
            print(f"❌ Video not found: {video_path} (Skipping...)")
            continue

        # Inner Loop: Iterate through all requested pipeline modes for this video
        for mode in PIPELINE_MODES:

            # ---> ADD THIS CHECK RIGHT HERE <---
            if (mode, video_name) in SKIP_LIST:
                print(f"⏭️ Skipping known bad combo: Mode [{mode}] for Video [{video_name}]")
                continue

            # Create a specific output directory for this mode
            output_dir = OUTPUT_BASE_DIR / f"sof_{mode}"
            
            print(f"\n==========================================")
            print(f"🔥 Starting Generation")
            print(f"   🎥 Video: {video_name}")
            print(f"   ⚙️  Mode: {mode}")
            print(f"   🏷️ Nudge: {CATEGORY_NUDGE if CATEGORY_NUDGE else 'None (General)'}")
            print(f"==========================================")

            # Initialize generator for this specific video/mode combo
            generator = BenchmarkGenerator(
                video_name=video_name,
                video_path=video_path,
                transcript_path=TRANSCRIPT_DIR,
                output_dir=output_dir
            )

            # Run the pipeline
            await generator.run_pipeline(
                pipeline_mode=mode, 
                target_count=TOTAL_QUESTIONS,
                category_nudge=CATEGORY_NUDGE
            )

    print(f"\n✅ Batch Processing Complete for {VIDEO_NAMES[0]} and other videos.")

if __name__ == "__main__":
    asyncio.run(main())
    print("MAKE SURE YOU'RE RUNNING THIS FROM Pupil/dataset_curation/generation OTHERWISE IT WONT WORK")