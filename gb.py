#!/usr/bin/env python

import logging
import argparse
import os.path
import itertools

from maroon import Model

from settings import settings
from explore import peek, sprawl, fixgis, graph
from predict import prep, fl
from base import gob
from base import utils

try:
    import networkx as nx
except ImportError:
    pass

try:
    import numpy
except ImportError:
    numpy = None

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Run gob jobs.')
    parser.add_argument('job',nargs='*')
    parser.add_argument('-s','--single',action="store_true",
                        help='run in a single process')
    parser.add_argument('-m','--mongo')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-i','--input')
    group.add_argument('--rm',action="store_true")
    group.add_argument('--force',action="store_true")
    group.add_argument('--dot',action="store_true")
    return parser.parse_args(argv)


def create_jobs(g):
    # the crawler
    g.add_source(utils.read_json, name='geotweets')
    g.add_map_job(sprawl.parse_geotweets,'geotweets',saver='split_save')
    g.add_map_job(sprawl.mloc_users,'parse_geotweets')
    g.add_map_job(sprawl.find_contacts,'mloc_users',
              reducer=gob.set_reduce)
    g.add_map_job(sprawl.contact_split,'find_contacts',saver='split_save')

    g.add_map_job(fixgis.gnp_gps,'mloc_users')
    g.add_cat('cat_gnp_gps','gnp_gps')
    g.add_map_job(fixgis.mdists,'cat_gnp_gps')
    g.add_map_job(sprawl.lookup_contacts,'contact_split',procs=15)
    g.add_map_job(sprawl.mloc_uids,'mloc_users')
    g.add_map_job(sprawl.trash_extra_mloc,'mloc_uids')
    g.add_map_job(sprawl.fix_mloc_mdists,'mloc_uids',requires=['mdists'])

    g.add_map_job(sprawl.pick_nebrs,'mloc_uids',
              requires=['lookup_contacts','mdists','trash_extra_mloc',
                        'fix_mloc_mdists'],
              reducer=gob.set_reduce,
              )
    g.add_map_job(sprawl.nebr_split, 'pick_nebrs', saver='split_save')
    g.add_map_job(sprawl.find_leafs,'nebr_split',reducer=gob.set_reduce)
    g.add_map_job(sprawl.contact_split,'find_leafs',
              name='leaf_split',saver='split_save')
    g.add_map_job(sprawl.saved_users,saver='split_save')
    g.add_map_job(sprawl.lookup_contacts, 'leaf_split',
              name='lookup_leafs',requires=['saved_users'])

    g.add_map_job(peek.contact_blur,'nebr_split',
                  requires=['lookup_leafs'],reducer=gob.avg_reduce)
    def nebr_clump(keys, clump):
        return keys[1]==clump

    g.add_clump(nebr_clump, clumps=["%02d"%x for x in xrange(100)],
                source='nebr_split', name='nebr_ids')

    g.add_map_job(peek.dirt_cheap_locals, 'nebr_ids',
              requires=['lookup_leafs'], procs=4,
              )
    g.add_map_job(peek.cheap_locals, 'nebr_ids',
              requires=['lookup_leafs'], procs=4,
              )
    g.add_map_job(peek.aint_cheap_locals, 'nebr_ids',
              requires=['lookup_leafs'], procs=4,
              )

    # the graphs
    g.add_map_job(peek.geo_ats,saver='split_save',requires=['find_leafs'])
    g.add_cat('cat_geo_ats','geo_ats')
    g.add_map_job(peek.at_tuples,'cat_geo_ats',saver='split_save')
    g.add_map_job(prep.pred_users,'mloc_uids')
    g.add_map_job(peek.geo_ated,'at_tuples',requires=['mloc_uids'])
    g.add_map_job(peek.edges_d,'pred_users',requires=['geo_ats'],procs=4)
    g.add_map_job(peek.edge_dists,'edges_d',reducer=gob.join_reduce)
    g.add_map_job(peek.rfrd_dists,'edges_d',
                  requires=['contact_blur','cheap_locals',
                            'dirt_cheap_locals','aint_cheap_locals'])
    g.add_cat('cat_rfrd_dists','rfrd_dists')
    g.add_map_job(peek.edge_leaf_dists,'edges_d')

    g.add_map_job(graph.graph_edge_types_cuml,'edge_dists')
    g.add_map_job(graph.graph_edge_types_prot,'edge_dists')
    g.add_map_job(graph.graph_edge_types_norm,'edge_dists')
    g.add_map_job(graph.graph_com_types,'edge_dists')
    g.add_map_job(graph.graph_edge_count,'cat_rfrd_dists')
    g.add_map_job(graph.graph_locals,'cat_rfrd_dists')
    g.add_cat('cat_edges_d','edges_d')
    g.add_map_job(graph.graph_local_groups,'cat_edges_d')

    g.add_map_job(peek.rfr_triads,'pred_users')
    g.add_cat('cat_rfr_triads','rfr_triads')
    g.add_map_job(graph.near_triads,'cat_rfr_triads')

    g.add_map_job(peek.rfr_indep,'pred_users')

    g.add_map_job(peek.diff_mloc_mdist,'mloc_uids')
    g.add_cat('cat_diff_mloc_mdist','diff_mloc_mdist')
    g.add_map_job(graph.graph_mloc_mdist,'cat_diff_mloc_mdist')


    # add noise to location field of geolocated users
    g.add_map_job(peek.contact_mdist,'contact_split')
    g.add_map_job(peek.contact_mdist,'mloc_uids',name='mloc_mdist')
    g.add_map_job(prep.mloc_blur, requires=['mloc_mdist','contact_mdist'])

    # fb predictor
    g.add_map_job(peek.contact_count,'contact_split',reducer=gob.sum_reduce)
    g.add_map_job(peek.mloc_tile,'mloc_uids',reducer=gob.join_reduce)
    g.add_map_job(peek.tile_split, 'mloc_tile', saver='split_save')
    g.add_map_job(peek.nebr_dists, 'tile_split')
    g.add_map_job(peek.stranger_dists, 'tile_split', requires=['contact_count'])
    g.add_map_job(peek.strange_nebr_bins, 'stranger_dists',
              name='strange_bins',reducer=gob.sum_reduce)
    g.add_map_job(peek.strange_nebr_bins, 'nebr_dists',
              name='nebr_bins',reducer=gob.sum_reduce)
    g.add_map_job( peek.contact_fit, (),
              requires=['strange_bins','nebr_bins'] )
    g.add_map_job(peek.lat_tile, saver='split_save')
    g.add_map_job(peek.stranger_prob, 'lat_tile',
              requires=['contact_count','contact_fit'])
    g.add_cat('stranger_prob_cat','stranger_prob')
    g.add_map_job(peek.stranger_mat,'stranger_prob_cat',encoding='npz')

    def train_set(keys, clump):
        return int(keys[0])//20!=int(clump)
    def eval_set(keys, clump):
        return int(keys[0])//20==int(clump)
    folds = [str(x) for x in xrange(5)]

    # prep
    g.add_map_job(prep.nebrs_d,'pred_users',
              requires=['mloc_blur','lookup_leafs','contact_blur'])
    g.add_clump(train_set, folds, 'nebrs_d', name='nebrs_train')
    g.add_clump(eval_set, folds, 'nebrs_d', name='nebrs_eval')

    # mdist_curves and utc_offset
    g.add_map_job(prep.mdist_real,'nebrs_d')
    g.add_clump(train_set, folds, 'mdist_real', name='mdist_train')
    g.add_map_job(prep.mdist_curves,'mdist_train')
    g.add_map_job(prep.utc_offset, 'nebrs_train')

    # the predictor
    g.add_map_job(fl.nebr_vect,'nebrs_d',requires=['dirt_cheap_locals'])
    g.add_clump(train_set, folds, 'nebr_vect', name='nvect_train')
    g.add_clump(eval_set, folds, 'nebr_vect', name='nvect_eval')

    g.add_map_job(fl.nebr_clf,'nvect_train',encoding='pkl')
    # FIXME : why don't we need vectors for nvect_eval?
    g.add_map_job(peek.vect_ratios, 'nvect_train',
              requires=['strange_bins','nebr_clf'] )
    g.add_map_job(peek.vect_fit, 'vect_ratios')
    g.add_map_job(graph.graph_vect_fit,'vect_fit')
    g.add_map_job(fl.predictions,'nebrs_eval',
              requires=['stranger_mat','mdist_curves','vect_fit','utc_offset','contact_fit'])
    g.add_cat('preds_cat','predictions')
    g.add_map_job(graph.gr_basic,'preds_cat')
    g.add_map_job(graph.gr_parts,'preds_cat')
    #g.add_map_job(graph.gr_count,'preds_cat')
    #g.add_map_job(graph.gr_usonly,'preds_us')
    g.add_map_job(fl.eval_preds,'predictions')

def make_gob(args):
    path = os.path.join(os.path.dirname(__file__),'data')
    if not args or args.single or args.input:
        env = gob.SimpleFileEnv(path,log_crashes=True)
    else:
        env = gob.MultiProcEnv(path,log_crashes=True,log_level=logging.INFO)
    my_gob = gob.Gob(env)
    create_jobs(my_gob)
    return my_gob


def setup(args):
    Model.database = utils.mongo(args.mongo or settings.region)
    logging.basicConfig(level=logging.INFO)
    if numpy:
        numpy.set_printoptions(precision=3, linewidth=160)


def inspect(job_name, source_set):
    """
    run a map reduce job on a single input and return its results.

    This is useful for poking around in ipython.
    """
    my_gob = make_gob(None)
    job = my_gob.jobs[job_name]
    funcs = job.runnable_funcs(my_gob.env)
    return my_gob.env.map_reduce_save(job, source_set, funcs)

def print_graph_dot(my_gob):
    srcs = ( (source.name,job)
             for job in my_gob.jobs
             for source in my_gob.jobs[job].sources
            )
    reqs = ( (req,job)
             for job in my_gob.jobs
             for req in my_gob.jobs[job].requires
            )
    dg = nx.DiGraph(itertools.chain(srcs,reqs))
    nx.write_dot(dg,'gob_deps.dot')


def run(my_gob,args):
    if args.dot:
        print_graph_dot(my_gob)
    if args.input:
        # just run the function for one input
        job = my_gob.jobs[args.job[0]]
        input_paths = [args.input.split(',')]
        my_gob.env.run(job,input_paths=input_paths)
    else:
        for cmd in args.job:
            if args.rm or args.force:
                my_gob.clear_job(cmd)
            if not args.rm:
                my_gob.run_job(cmd)

def main(*argv):
    # set globals so that we can %run this in ipython
    global my_gob, env
    args = parse_args(argv or None)
    my_gob = make_gob(args)
    setup(args)
    run(my_gob,args)
    env = my_gob.env


if __name__ == '__main__':
    main()
