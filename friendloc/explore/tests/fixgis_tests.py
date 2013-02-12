from friendloc.base.tests import SimpleGobTest


class TestSprawlToContacts(SimpleGobTest):

    def test_mdists(self):
        btx = dict(lat=31,lng=-96,code="PPLA2",fid=17)
        cstx = dict(lat=30,lng=-96,code="PPLA2",fid=32)
        self.FS['gnp_gps.03'] = [
            (btx,[-96,32],),
            (btx,[-96,32],),
            (btx,[-96,31],),
        ]
        self.FS['gnp_gps.04'] = [
            (cstx,[-96,30],),
        ]
        self.gob.run_job('mdists')
        mdists = self.FS['mdists'][0]
        self.assertEqual(mdists['other'],0)
        self.assertAlmostEqual(mdists['17'],69.0976,places=3)
        self.assertEqual(len(mdists),2)
