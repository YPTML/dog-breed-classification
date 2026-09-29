"""Strict public API assertions shared by synthetic-input regression tests."""
import math


def assert_three_state_response(test, data, class_names):
    state = data['decision']
    test.assertIn(state, {'DOG', 'UNCERTAIN', 'NOT_DOG'})
    test.assertEqual(data['status'], state)
    test.assertEqual(data['final_status'], state)
    test.assertIs(
        data['is_dog'],
        True if state == 'DOG' else (False if state == 'NOT_DOG' else None),
    )
    if state != 'DOG':
        test.assertTrue(data['message'])
    top1, top5 = data['prediction'], data['top5']
    test.assertIsInstance(top1, dict)
    test.assertEqual(len(top5), 5)
    test.assertEqual(len({item['index'] for item in top5}), 5)
    test.assertEqual(len({item['breed'] for item in top5}), 5)
    previous = 1.0
    for item in top5:
        test.assertIsInstance(item['index'], int)
        test.assertIn(item['index'], range(120))
        test.assertEqual(item['breed'], class_names[item['index']])
        test.assertTrue(math.isfinite(item['confidence']))
        test.assertGreaterEqual(item['confidence'], 0)
        test.assertLessEqual(item['confidence'], previous)
        previous = item['confidence']
    test.assertEqual(top1['breed'], top5[0]['breed'])
    test.assertEqual(top1['confidence'], top5[0]['confidence'])
