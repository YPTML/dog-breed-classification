import unittest
import numpy as np
from src.ood.calibrate_v1_1 import gap_table, rates


class RevisionTests(unittest.TestCase):
    def test_midpoint_metrics_match_actual_partitions(self):
        ids=np.array([.1,.2,.2,.4])
        ood=np.array([.2,.3,.5])
        for row in gap_table(ids,ood):
            t=row['threshold']
            self.assertNotIn(t,np.r_[ids,ood])
            self.assertEqual(row['tp'],int((ood>=t).sum()))
            self.assertEqual(row['fp'],int((ids>=t).sum()))

    def test_three_state_boundaries_and_gate(self):
        policy={'uncertain_threshold':.2,'ood_threshold':.4,'breed_confidence_threshold':None}
        scores=np.array([.1,.2,.3,.4])
        p=np.array([[.6,.4]]*4)
        self.assertEqual(rates(scores,p,policy),{'DOG':.25,'UNCERTAIN':.5,'NOT_DOG':.25})
        policy['breed_confidence_threshold']=.7
        self.assertEqual(rates(scores,p,policy),{'DOG':0.,'UNCERTAIN':.75,'NOT_DOG':.25})


if __name__=='__main__':unittest.main()
