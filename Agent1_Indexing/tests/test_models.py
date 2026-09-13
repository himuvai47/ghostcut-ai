import unittest

from indexing_agent.models import validate_vision_payload


class VisionPayloadTests(unittest.TestCase):
    def good(self):
        return {
            "description": "Macro view of mechanical watch gears and metallic movement components.",
            "primary_subject": "mechanical watch movement",
            "secondary_subjects": [],
            "primary_product": "mechanical wristwatch movement",
            "secondary_products": [],
            "objects": ["gears"],
            "people": [],
            "actions": ["gears moving"],
            "environment": "product macro setup",
            "shot_type": "macro",
            "camera_motion": "unknown",
            "content_type": "product_detail",
            "visual_attributes": ["metallic"],
            "visual_concepts": ["precision", "craftsmanship"],
            "visible_text": [],
            "observed_details": [
                "Multiple gears are visible.",
                "Metallic components fill most of the frame.",
                "The movement is shown at very close range.",
                "The watch mechanism is the dominant subject.",
            ],
            "uncertain_inferences": [],
            "brand": "unknown",
            "model_name": "unknown",
            "subject_focus": "movement",
            "shot_angle": "front",
            "editorial_role": "movement_detail",
            "visual_energy": "calm",
            "lighting_style": "neutral",
            "composition": "detail_only",
            "quality_flags": [],
            "usable": True,
            "exclusion_reason": "none",
        }

    def test_valid_payload_derives_confidence(self):
        result = validate_vision_payload(self.good())
        self.assertEqual(result.shot_type, "macro")
        self.assertTrue(result.usable)
        self.assertIn(result.confidence_level, {"medium", "high"})
        self.assertGreaterEqual(result.analysis_confidence, 0.64)
        self.assertTrue(result.confidence_basis)

    def test_missing_field_rejected(self):
        payload = self.good()
        del payload["objects"]
        with self.assertRaises(ValueError):
            validate_vision_payload(payload)

    def test_invalid_enum_rejected(self):
        payload = self.good()
        payload["shot_type"] = "The camera is stationary and close to the watch."
        with self.assertRaises(ValueError):
            validate_vision_payload(payload)

    def test_visible_text_placeholder_removed(self):
        payload = self.good()
        payload["visible_text"] = ["No visible text in the frames.", "ROLEX"]
        result = validate_vision_payload(payload)
        self.assertEqual(result.visible_text, ["ROLEX"])

    def test_uncertain_observation_is_moved(self):
        payload = self.good()
        payload["observed_details"] = ["The item is likely a luxury watch.", "The blue dial is visible."]
        result = validate_vision_payload(payload)
        self.assertEqual(result.observed_details, ["The blue dial is visible."])
        self.assertIn("The item is likely a luxury watch.", result.uncertain_inferences)

    def test_direct_observation_recovers_from_inference(self):
        payload = self.good()
        payload["uncertain_inferences"] = ["The camera remains static across the frames."]
        result = validate_vision_payload(payload)
        self.assertIn("The camera remains static across the frames.", result.observed_details)
        self.assertEqual(result.uncertain_inferences, [])

    def test_unhedged_interpretation_is_marked_uncertain(self):
        payload = self.good()
        payload["uncertain_inferences"] = ["The footage is part of a professional product review."]
        result = validate_vision_payload(payload)
        self.assertEqual(result.uncertain_inferences, ["Possibly: The footage is part of a professional product review."])

    def test_end_screen_forced_unusable_and_text_screen(self):
        payload = self.good()
        payload["content_type"] = "end_screen"
        payload["shot_type"] = "medium"
        payload["usable"] = True
        payload["exclusion_reason"] = "none"
        result = validate_vision_payload(payload)
        self.assertFalse(result.usable)
        self.assertEqual(result.exclusion_reason, "end_screen")
        self.assertEqual(result.shot_type, "text_screen")

    def test_entities_are_enriched(self):
        payload = self.good()
        payload.update({
            "primary_subject": "A person wearing white gloves",
            "primary_product": "Rolex wristwatch",
            "secondary_products": ["another wristwatch"],
            "objects": [],
            "people": [],
            "shot_type": "close_up",
            "actions": ["The person rotates the watch."],
        })
        result = validate_vision_payload(payload)
        self.assertIn("Rolex wristwatch", result.objects)
        self.assertIn("another wristwatch", result.objects)
        self.assertIn("A person wearing white gloves", result.people)

    def test_product_only_medium_box_shot_becomes_close_up(self):
        payload = self.good()
        payload.update({
            "description": "A Rolex watch is displayed inside an open green presentation box.",
            "primary_subject": "Rolex watch",
            "primary_product": "Rolex watch",
            "objects": ["green presentation box"],
            "people": [],
            "actions": [],
            "environment": "open green watch box",
            "shot_type": "medium",
            "content_type": "product_detail",
            "observed_details": ["The watch is inside an open green box."],
        })
        result = validate_vision_payload(payload)
        self.assertEqual(result.content_type, "product_in_box")
        self.assertEqual(result.shot_type, "close_up")

    def test_presenter_demo_classification(self):
        payload = self.good()
        payload.update({
            "primary_subject": "A person",
            "primary_product": "wristwatch",
            "people": ["A person"],
            "actions": ["The person speaks while holding the watch."],
            "shot_type": "medium",
            "content_type": "product_detail",
        })
        result = validate_vision_payload(payload)
        self.assertEqual(result.content_type, "presenter_demo")

    def test_product_handling_classification(self):
        payload = self.good()
        payload.update({
            "primary_subject": "Gloved hands",
            "primary_product": "silver wristwatch with blue dial",
            "people": ["Gloved hands"],
            "actions": ["The hands rotate the watch to show different angles."],
            "shot_type": "close_up",
            "content_type": "product_detail",
        })
        result = validate_vision_payload(payload)
        self.assertEqual(result.content_type, "product_handling")

    def test_generic_product_gets_lower_confidence_than_specific_supported_product(self):
        generic = self.good()
        generic.update({
            "primary_subject": "A person",
            "primary_product": "A watch",
            "people": ["A person"],
            "shot_type": "medium",
            "visible_text": [],
            "observed_details": ["A person holds a watch."],
            "uncertain_inferences": ["The person is likely presenting the watch."],
        })
        specific = self.good()
        specific.update({
            "primary_product": "Rolex Oyster Perpetual wristwatch",
            "visible_text": ["ROLEX", "OYSTER PERPETUAL"],
        })
        a = validate_vision_payload(generic)
        b = validate_vision_payload(specific)
        self.assertLess(a.analysis_confidence, b.analysis_confidence)
        self.assertEqual(a.confidence_level, "medium")
        self.assertEqual(b.confidence_level, "high")
