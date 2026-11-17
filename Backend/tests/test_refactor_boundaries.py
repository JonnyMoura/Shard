import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evolutionary_module
import feedback_validation
import generation_payload
import synthesis_pipeline
from fitness import automatic_fitness


class FitnessTests(unittest.TestCase):
    def test_category_fitness_values_are_preserved(self):
        descriptors = [1300, 3200, 1500, 5, 0.08]
        expected = {
            "low": 0.6839983137345791,
            "mid": 0.7733078688181331,
            "high": 0.6125588986868339,
            "rhythmic": 0.7414433193244881,
        }

        for category, expected_score in expected.items():
            with self.subTest(category=category):
                self.assertAlmostEqual(
                    automatic_fitness(descriptors, category),
                    expected_score,
                    places=12,
                )

    def test_evolution_module_continues_to_export_fitness(self):
        self.assertIs(evolutionary_module.automatic_fitness, automatic_fitness)
        self.assertEqual(len(evolutionary_module.emitters), 16)


class SynthesisPipelineTests(unittest.TestCase):
    def test_source_path_resolution_checks_expected_locations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            source_dir = root / "src"
            source_dir.mkdir()
            source_path = source_dir / "sample.wav"
            source_path.touch()

            self.assertEqual(
                synthesis_pipeline.resolve_source_path(
                    "sample.wav", str(data_dir), str(source_dir)
                ),
                str(source_path),
            )

    def test_empty_job_batch_returns_without_spawning_workers(self):
        self.assertEqual(synthesis_pipeline.run_synthesis_jobs([]), [])

    def test_synthesis_job_omits_audio_when_requested(self):
        job = {
            "solution": [1.0],
            "source_tracks": ["track.wav"],
            "category": "low",
            "duration": 2,
            "include_audio": False,
        }
        with (
            mock.patch.object(
                synthesis_pipeline,
                "generate_audio_stream",
                return_value=("audio", 44100),
            ),
            mock.patch.object(
                synthesis_pipeline,
                "extract_audio_descriptors",
                return_value=["descriptors"],
            ),
        ):
            result = synthesis_pipeline.synthesize_job(job)

        self.assertEqual(result, (None, 44100, ["descriptors"], ["track.wav"]))

    def test_preview_reuses_original_descriptors_and_sources(self):
        jobs = [{"duration": 2, "include_audio": False}]
        results = [("old-audio", 22050, ["original"], ["source.wav"])]
        preview = ("preview-audio", 44100, ["preview"], ["other-source.wav"])

        with mock.patch.object(
            synthesis_pipeline, "run_synthesis_jobs", return_value=[preview]
        ):
            synthesis_pipeline.render_selected_previews([0], jobs, results)

        self.assertEqual(
            results,
            [("preview-audio", 44100, ["original"], ["source.wav"])],
        )
        self.assertEqual(jobs, [{"duration": 2, "include_audio": False}])


class FeedbackValidationTests(unittest.TestCase):
    def test_feedback_for_just_evaluated_generation_is_accepted(self):
        feedback = {
            "generation": 7,
            "feedback": [
                {"solution_id": "solution_007_000"},
                {"solution_id": "solution_007_001"},
            ],
        }

        self.assertTrue(
            feedback_validation.feedback_matches_generation(feedback, 7, 2)
        )

    def test_stale_generation_and_duplicate_positions_are_rejected(self):
        stale_feedback = {
            "generation": 6,
            "feedback": [{"solution_id": "solution_006_000"}],
        }
        duplicate_feedback = {
            "generation": 7,
            "feedback": [
                {"solution_id": "solution_007_000"},
                {"solution_id": "solution_007_000"},
            ],
        }

        self.assertFalse(
            feedback_validation.feedback_matches_generation(stale_feedback, 7, 1)
        )
        self.assertFalse(
            feedback_validation.feedback_matches_generation(duplicate_feedback, 7, 2)
        )


class GenerationPayloadTests(unittest.TestCase):
    def test_payload_keeps_generation_ids_and_audio_metadata(self):
        solution = list(range(11))
        result = (
            [0.0, 0.5],
            2,
            [100, 200, 300, 4, 0.1],
            ["C:\\audio\\track.wav"],
        )

        with mock.patch("builtins.print"):
            payload = generation_payload.build_generation_payload(
                7,
                [solution],
                [result],
                [0.8],
                ["mid"],
                {0},
                lambda _: "low",
                lambda _audio, _sample_rate: "encoded-audio",
            )

        self.assertEqual(payload["generation"], 7)
        self.assertEqual(payload["solutions"][0]["solution_id"], "solution_007_000")
        self.assertEqual(payload["solutions"][0]["audio"]["data"], "encoded-audio")
        self.assertEqual(payload["solutions"][0]["audio"]["duration"], 1.0)
        self.assertEqual(payload["solutions"][0]["sources"], ["track.wav"])
        self.assertEqual(payload["statistics"]["inherited_count"], 1)


if __name__ == "__main__":
    unittest.main()
