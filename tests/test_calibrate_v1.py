"""Boundary and convention regression tests for the guarded V1 protocol."""
import unittest
import numpy as np
from src.ood.calibrate_v1 import threshold_table, strategies, score_methods


class CalibrationV1Tests(unittest.TestCase):
    def test_ties_are_rejected_and_extremes_present(self):
        table = threshold_table(np.array([.1,.5]),np.array([.5,.9]))
        tied = next(r for r in table if r['threshold']==.5)
        self.assertEqual((tied['tp'],tied['fp'],tied['fn'],tied['tn']),(2,1,0,1))
        self.assertEqual(table[0]['recall'],1)
        self.assertEqual(table[-1]['id_acceptance'],1)
        self.assertEqual(table[-1]['ood_far'],1)

    def test_strategies_preserve_id_constraint(self):
        table=threshold_table(np.arange(100)/100,np.array([.2,.8,1.,2.]))
        chosen=strategies(table)
        self.assertGreaterEqual(chosen['B_id_acceptance_95']['id_acceptance'],.95)
        self.assertEqual(chosen['A_max_ood_f1']['f1'],max(r['f1'] for r in table))
        self.assertEqual(chosen['C_weighted_cost_3_to_1']['weighted_cost'],min(r['weighted_cost'] for r in table))

    def test_energy_retains_true_logit_offset(self):
        p=np.full((2,120),1/120)
        logits=np.vstack([np.zeros(120),np.full(120,10.)])
        d=dict(probabilities=p,logits=logits,embeddings=np.ones((2,1280)))
        s=score_methods(d,np.ones((120,1280)))
        self.assertTrue(np.allclose(s['entropy'],1))
        self.assertTrue(np.allclose(s['msp'],119/120))
        self.assertAlmostEqual(s['energy_T1'][0]-s['energy_T1'][1],10.)


if __name__=='__main__': unittest.main()
