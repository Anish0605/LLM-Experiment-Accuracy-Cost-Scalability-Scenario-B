import unittest

import app


def canonical_record():
    return {
        "customer_age": 42,
        "product_price": 19.99,
        "discount_percent": 12.5,
        "product_rating": 4.25,
        "past_purchase_count": 7,
        "past_return_rate": 0.125,
        "delivery_delay_days": 2.5,
        "session_length_minutes": 23.75,
        "num_product_views": 14,
        "device_type": "mobile",
        "product_category": "clothing",
        "shipping_method": "same_day",
        "payment_method": "apple_pay",
        "used_coupon": 1,
        "returned": 0,
        "order_id": "123456",
    }


class ScenarioBTests(unittest.TestCase):
    def test_alias_aware_rules_restore_ground_truth_across_multiple_patterns(self):
        gold = canonical_record()
        for row_index in range(30):
            with self.subTest(row_index=row_index):
                altered = app.corrupt_scenario_b(gold, row_index)
                result = app.alias_aware_transform(altered)
                for field in app.FIELDS:
                    self.assertTrue(
                        app.equal_value(field, result.get(field), gold[field]),
                        msg=f"Row index {row_index}: {field} was {result.get(field)!r}, expected {gold[field]!r}",
                    )

    def test_frozen_rules_do_not_silently_handle_scenario_b_schema_aliases(self):
        gold = canonical_record()
        altered = app.corrupt_scenario_b(gold, 1)
        frozen = app.rule_transform(altered)
        comparison = app.evaluate({"R00001": gold}, {"R00001": frozen}, ["R00001"], "Frozen")
        self.assertLess(comparison["field_accuracy_pct"], 100.0)

    def test_scenario_b_inputs_have_all_sixteen_fields_after_alias_aware_mapping(self):
        gold = canonical_record()
        altered = app.corrupt_scenario_b(gold, 7)
        result = app.alias_aware_transform(altered)
        self.assertEqual(set(result.keys()), set(app.FIELDS))
        self.assertTrue(all(value is not None for value in result.values()))


if __name__ == "__main__":
    unittest.main()
