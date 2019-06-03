import codecs
import multiprocessing
import operator
import random
import tempfile

import networkx as nx
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from mednorm.pmt_helpers import import_pymedtermino

pymedtermino = import_pymedtermino()

from collections import Counter

try:
    from pymedtermino.meddra import MEDDRA
except:
    MEDDRA = None
try:
    from pymedtermino.snomedct import SNOMEDCT
except:
    SNOMEDCT = None

import time
from mednorm.ontology.mapping import BioPortalTerminologyMapper
from mednorm.ontology.mapping import Terminology
from mednorm.tweetnlp.tokenize import word_tokenize
from functools import partial
from tqdm import tqdm
from mednorm.utils import makedirs_file
from multiprocessing import Pool
from itertools import combinations, permutations

MEDDRA_HIER_ORDER = ['LLT', 'PT', 'HLT', 'HLGT', 'SOC']


def get_concept_pair_key(c1, c2):
    a = min(c1['concept_id'], c2['concept_id'])
    b = max(c1['concept_id'], c2['concept_id'])
    return "%s_%s" % (a, b)


class GensimEpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.previous = None
        self.pbar = None

    def on_train_begin(self, model):
        model.compute_loss = True

    def on_batch_end(self, model):
        self._update_metrics(model)

    def on_epoch_begin(self, model):
        if self.pbar is None:
            self.pbar = tqdm(desc='training', total=model.epochs)
        self.previous = model

    def _update_metrics(self, model):
        self.pbar.set_postfix(loss=model.running_training_loss,
                              min_alpha=model.min_alpha_yet_reached)

    def on_epoch_end(self, model):
        self.epoch += 1
        self._update_metrics(model)
        self.pbar.update()

    def on_train_end(self, model):
        self.pbar.close()


class Node(object):
    def __init__(self, **properties):
        self._node_id = properties.pop('node_id', None)
        self.properties = properties

    def get_node_id(self):
        if self._node_id is None:
            raise ValueError("node id is not defined!")
        return self._node_id

    def set_node_id(self, node_id):
        self._node_id = node_id

    def get(self, item, default=None):
        try:
            return self.__getitem__(item)
        except Exception:
            return default

    def __getitem__(self, item):
        if item == 'node_id':
            return self.get_node_id()
        return self.properties.get(item, None)

    def __unicode__(self):
        return u"%s" % self.get_node_id()

    def __str__(self):
        return "%s" % self.get_node_id()

    def __hash__(self):
        return hash(self.get_node_id())

    def __eq__(self, other):
        return hash(self) == hash(other)


class RootNode(Node):
    def __init__(self, name='ROOT_NODE'):
        super(RootNode, self).__init__(name=name, root=1)

    def get_node_id(self):
        return "ROOT_NODE"


class DatasetNode(Node):
    def __init__(self, dataset_name):
        super(DatasetNode, self).__init__(dataset_name=dataset_name)

    def get_node_id(self):
        return "dataset_%s" % self['dataset_name'].lower()


class OntoConceptNode(Node):
    def __init__(self, concept_id, origin_terminology, term, **props):
        super(OntoConceptNode, self).__init__(concept_id=str(concept_id),
                                              origin_terminology=origin_terminology,
                                              term=term, **props)

    def get_node_id(self):
        return "%s_%s" % (self['origin_terminology'], self['concept_id'])


class NameNode(Node):
    def __init__(self, name):
        super(NameNode, self).__init__(name=name)

    def get_node_id(self):
        return self['name']


class InstanceNode(Node):
    def __init__(self, instance_id):
        super(InstanceNode, self).__init__(instance_id=instance_id)

    def get_node_id(self):
        return self['instance_id']


class PhraseNode(Node):
    def __init__(self, phrase):
        super(PhraseNode, self).__init__(phrase=phrase)

    def get_node_id(self):
        return self['phrase']


class TokenNode(Node):
    def __init__(self, token):
        super(TokenNode, self).__init__(token=token)

    def get_node_id(self):
        return "_%s_" % self['token']


def get_meddra_props(concept_id):
    try:
        c = MEDDRA.get_by_meddra_code(concept_id)[0]
        code = c.code
        return {
            'hier_level': code[:code.index("_")],
            'active': bool(c.active),
        }
    except Exception:
        raise ValueError("Unable to get meddra parents for %s" % concept_id)


def get_snomed_props(concept_id):
    try:
        c = SNOMEDCT.get(concept_id)
        return {
            'active': bool(c.active),
        }
    except Exception:
        raise ValueError("Unable to get snomed props for %s" % concept_id)


def get_term_func(terminology):
    return partial(Terminology.get_term, terminology=terminology)


def get_parents_func(terminology):
    return partial(Terminology.get_parents, terminology=terminology)


def get_props_func(term_name):
    if term_name == Terminology.MEDDRA:
        return get_meddra_props
    elif term_name == Terminology.SNOMEDCT:
        return get_snomed_props

    return None


def get_node_edges(mdg, node, edge_name):
    return [(u, v, d) for u, v, d in mdg.edges(node, data=True)
            if d.get('name', None) == edge_name]


def get_recursive_nodes(mdg, node, edge_name, depth=0, nodes=None,
                        curr_depth=0, verbose=0):
    if nodes is None:
        nodes = {node}
    if 0 < depth < curr_depth:
        return nodes
    if verbose:
        print("%s%d.\t%s" % (" " * (curr_depth * 2), curr_depth,
                             node.get('term', str(node))))
    for u, v, d in get_node_edges(mdg, node, edge_name):
        if v in nodes:
            continue
        nodes.add(v)
        nodes.update(get_recursive_nodes(mdg, v, edge_name, depth=depth,
                                         nodes=nodes,
                                         curr_depth=curr_depth + 1,
                                         verbose=verbose))
    return nodes


def get_node_in_edges(mdg, node, edge_name):
    return [(u, v, d) for u, v, d in mdg.in_edges(node, data=True)
            if d.get('name', None) == edge_name]


def get_recursive_nodes_reverse(mdg, node, edge_name, depth=0, nodes=None,
                                curr_depth=0):
    if nodes is None:
        nodes = {node}

    if 0 < depth < curr_depth:
        return nodes

    for u, v, d in get_node_in_edges(mdg, node, edge_name):
        if u in nodes:
            continue
        nodes.add(u)
        nodes.update(
            get_recursive_nodes_reverse(mdg, u, edge_name, depth=depth,
                                        nodes=nodes, curr_depth=curr_depth + 1))
    return nodes


def get_concept_closest_meddra_hier(mdg, concept_node, hier_level='PT',
                                    max_depth=0):
    chl = concept_node.get('hier_level', None)
    if chl == hier_level:
        return {concept_node}
    elif (chl in MEDDRA_HIER_ORDER and MEDDRA_HIER_ORDER.index(
            chl) > MEDDRA_HIER_ORDER.index(hier_level)):
        # print("concept node level (%s) is higher than %s" % (
        #     chl, hier_level
        # ))
        return set()
    elif (chl in MEDDRA_HIER_ORDER and MEDDRA_HIER_ORDER.index(
            chl) < MEDDRA_HIER_ORDER.index(hier_level)):
        result = set()
        parents = get_recursive_nodes(mdg, concept_node, "IS_A",
                                      depth=max_depth)
        for p in parents:
            if p == concept_node:
                continue
            phl = p.get('hier_level', None)
            if phl == hier_level:
                result.add(p)

    # mappings = get_recursive_nodes(mdg, concept_node, "MAPPED_TO",
    #                                depth=max_depth)
    # for m in mappings:
    #     mhl = m.get('hier_level', None)
    #     if mhl == hier_level:
    #         result.add(m)
    #     elif (mhl in MEDDRA_HIER_ORDER and MEDDRA_HIER_ORDER.index(
    #             mhl) < MEDDRA_HIER_ORDER.index(hier_level)):
    #         parents = get_recursive_nodes(mdg, m, "IS_A",
    #                                       depth=max_depth)
    #         for p in parents:
    #             if p == concept_node or p == concept_node:
    #                 continue
    #             phl = p.get('hier_level', None)
    #             if phl == hier_level:
    #                 result.add(p)
    return result


def get_instance_meddra_pt(mdg, instance_node, max_depth=0):
    result = set()
    annotations = [v for _, v, _ in get_node_edges(mdg, instance_node,
                                                   "ANNOTATED_AS")]
    for a in annotations:
        if a.get('hier_level', None) == 'PT':
            result.add(a)
        elif a.get('hier_level', None) == 'LLT':
            p = [v for _, v, _ in get_node_edges(mdg, a, "IS_A")
                 if v != a]
            result.update(p)

    for a in annotations:
        mappings = get_recursive_nodes(mdg, a, "MAPPED_TO", depth=max_depth)
        for m in mappings:
            if m.get('hier_level', None) == 'PT':
                result.add(m)
            elif m.get('hier_level', None) == 'LLT':
                p = [v for _, v, _ in get_node_edges(mdg, m, "IS_A")
                     if v != m]
                result.update(p)
    return result


def get_term_codes(mdg, all_mappings, all_annotations):
    # TODO: Would be nice to have a stats for skipped entities
    meddra_codes = set()
    sct_ids = set()
    final_mappings = set()
    active_mappings = [m for m in all_mappings if m['active']]
    for m in active_mappings:
        # skip/ignore UMLS mappings
        if m.get(
                'origin_terminology') == Terminology.UMLS_CUI.value:
            continue
        # skip non-(finding/disorder) SCT mappings
        if m.get(
                'origin_terminology') == Terminology.SNOMEDCT.value:
            parents = get_recursive_nodes(mdg, m, "IS_A")
            if not any(p['concept_id'] == '404684003' for p in parents):
                continue

        # skip mappings from lower levels

        # if m.get('origin_terminology') == Terminology.MEDDRA.value:
        #     if m.get('hier_level', None) not in ('PT', 'LLT'):
        #         pts = [mm for mm in
        #                self.get_recursive_nodes_reverse(m, "IS_A")
        #                if  mm != m
        #         and mm.get('hier_level', None) == 'PT']
        #         final_mappings.update(pts)
        #         continue

        if any((mm in all_mappings and mm != m
                and mm.get('hier_level', None) in ('LLT',))
               for mm in get_recursive_nodes(mdg, m, "IS_A")):
            continue
        # for MEDDRA LLT, always get PT
        if m.get('hier_level', None) == 'LLT':
            p = [v for _, v, _ in get_node_edges(mdg, m, "IS_A")
                 if v != m]
            final_mappings.update(p)
            continue
        final_mappings.add(m)

    # filter mappings that have not been used in annotations,
    # but skip if only one mapping is provided
    filtered_mappings = set()
    all_terminologies = set(m.get('origin_terminology') for m in final_mappings)
    for term in all_terminologies:
        term_mappings = [m for m in final_mappings if m.get(
            'origin_terminology') == term]
        if len(term_mappings) > 1:
            flt_term_mappings = [m for m in term_mappings if m in
                                 all_annotations]
            if len(flt_term_mappings) > 1:
                term_mappings = flt_term_mappings
        filtered_mappings.update(term_mappings)

    final_mappings = list(filtered_mappings)

    # print("Mappings:")
    for m in final_mappings:
        is_orig = m in all_annotations

        if m.get('origin_terminology') == Terminology.MEDDRA.value:
            if not is_orig and m['hier_level'] not in ('LLT', 'PT'):
                print("  check ", m['hier_level'], m['concept_id'])
                for c in get_recursive_nodes_reverse(mdg, m, "IS_A"):
                    print("      ", c['concept_id'])

                if any(c in final_mappings for c in
                       get_recursive_nodes_reverse(mdg, m,
                                                   "IS_A")
                       if m != c):
                    continue

            meddra_codes.add(m['concept_id'])
        elif m.get('origin_terminology') == \
                Terminology.SNOMEDCT.value:
            if (not is_orig and
                    any(c in final_mappings for c in
                        get_recursive_nodes_reverse(mdg, m, "IS_A")
                        if m != c and not mdg.has_edge(m, c, "IS_A"))):
                # print("SKIP %s !!" % m)
                continue

            sct_ids.add(m['concept_id'])
        # print(m)
    # print("------------")
    return meddra_codes, sct_ids


def process_node(node, mdg, all_annotations, max_depth=0):
    print("------- %s -------" % node['instance_id'])
    dataset = [u for u, _, _ in
               get_node_in_edges(mdg, node, "HAS_INSTANCE")][0]
    # print node['instance_id']
    phrase = [v for _, v, _ in get_node_edges(mdg, node,
                                              "DESCRIBED_AS")][0]
    annotations = [v for _, v, _ in get_node_edges(mdg, node,
                                                   "ANNOTATED_AS")]
    # print phrase
    # print("Annotations:")
    all_mappings = set()
    for ann in annotations:
        # print ann
        all_mappings.update(get_recursive_nodes(mdg, ann, "MAPPED_TO",
                                                depth=max_depth))
    # print("------------")
    pt_nodes = get_instance_meddra_pt(mdg, node, max_depth=max_depth)
    # print("PTs: %s" % pt_nodes)
    print([n['concept_id'] for n in pt_nodes])
    meddra_codes = set()
    sct_ids = set()

    if pt_nodes:
        for pt_node in pt_nodes:
            mappings = get_recursive_nodes(mdg, pt_node, "MAPPED_TO")
            all_mappings.update(mappings)
        meddra_codes, sct_ids = get_term_codes(mdg, all_mappings,
                                               all_annotations)
    else:
        print("UNABLE TO FIND PT NODE for %s!" % node)

    # # filter meddra_codes
    # flt_meddra_codes = [m for m in meddra_codes
    #                     if m not in all_annotations]
    # if len(flt_meddra_codes) > 0:
    #     meddra_codes = flt_meddra_codes
    #
    # # filter sct_ids
    # flt_sct_ids = [m for m in sct_ids
    #                if m not in all_annotations]
    # if len(flt_sct_ids) > 0:
    #     sct_ids = flt_sct_ids
    print(node['instance_id'], meddra_codes, sct_ids)
    return (node['instance_id'], {
        'meddra_codes': meddra_codes,
        'sct_ids': sct_ids
    })


def get_keyed_annotations_distance(a_b, g):
    return get_concept_pair_key(*a_b), get_annotations_distance(a_b, g)


# TODO: Add comments
def get_annotations_distance(a_b, g):
    a, b = a_b
    try:
        return nx.shortest_path_length(g, source=a, target=b, weight='weight')
    except nx.exception.NetworkXNoPath:
        print("NO PATH BETWEEN %s and %s" % (a['term'], b['term']))
        return np.inf


def get_keyed_emb_annotations_distance(a_b, emb):
    return get_concept_pair_key(*a_b), get_emb_annotations_distance(a_b, emb)


def get_emb_annotations_distance(a_b, emb):
    a, b = a_b
    try:
        return emb.distance(a['node_id'], b['node_id'])
    except KeyError as e:
        print("No distance %s and %s (err: %s)" % (a['term'], b['term'], e))
        return np.inf


def add_reverse_edges(g, target_edges):
    reverse = []
    for (u, v, rel) in g.edges(data=True):
        if rel['name'] not in target_edges:
            continue
        rev = {'name': 'REVERSE_%s' % rel['name']}
        reverse.append((v, u, rev))

    g.add_edges_from(reverse)


class MedNormOntology(object):
    def __init__(self):
        self.mdg = nx.MultiDiGraph()
        self.udg = None
        self.udg_nodemap = None
        self.graph_embeddings = None
        self.__ann_distances = {}
        self.__cg = None
        self.__cg_calc = None

    def match_nodes(self, node_cls, **kwargs):
        for node in self.mdg.nodes():
            if not isinstance(node, node_cls):
                continue
            match_cnt = 0
            for kw, kv in kwargs.items():
                if node.get(kw, None) in kv:
                    match_cnt += 1
            if match_cnt == len(kwargs.items()):
                yield node

    def match_concept_nodes(self, concept_ids):
        return self.match_nodes(OntoConceptNode, concept_id=concept_ids)

    def find_node(self, node_cls, **kwargs):
        return next(self.match_nodes(node_cls, **kwargs), None)

    def add_node(self, node_cls, **kwargs):
        n = node_cls(**kwargs)
        if n in self.mdg.nodes:
            return n, False
        self.mdg.add_node(n)
        return n, True

    def add_relation(self, u, v, rel, **props):
        self.mdg.add_edge(u, v, rel, name=rel, **props)

    def add_dataset(self, dataset_name):
        n, added = self.add_node(DatasetNode, dataset_name=dataset_name)
        return n

    def add_instance(self, instance_id, phrase):
        inst_node, added = self.add_node(InstanceNode, instance_id=instance_id)
        if added:
            phrase_node = self.add_phrase(phrase)
            self.add_relation(inst_node, phrase_node, "DESCRIBED_AS")
        return inst_node

    def add_concept(self, concept_id, origin_terminology,
                    term_func, parents_func=None, props_func=None):
        props = {}
        if props_func:
            props = props_func(concept_id)

        term = term_func(concept_id)
        concept_node, added = self.add_node(
            OntoConceptNode,
            concept_id=concept_id,
            origin_terminology=origin_terminology.value,
            term=term,
            **props)

        if added:
            name_node = self.add_name(term)
            self.add_relation(concept_node, name_node, "NAMED_AS")
            if parents_func:
                parents = parents_func(concept_id)
                for p in parents:
                    p_node = self.add_concept(p, origin_terminology,
                                              term_func, parents_func,
                                              props_func=props_func)
                    self.add_relation(concept_node, p_node, "IS_A")
        return concept_node

    def add_phrase(self, phrase):
        phrase_node, added = self.add_node(PhraseNode, phrase=phrase)
        if added:
            tokens = word_tokenize(phrase)
            for t in tokens:
                token_node = self.add_token(token=t)
                self.add_relation(phrase_node, token_node, "CONTAINS")
        return phrase_node

    def add_name(self, name):
        name_node, added = self.add_node(NameNode, name=name)
        if added:
            tokens = word_tokenize(name)
            for t in tokens:
                token_node = self.add_token(token=t)
                self.add_relation(name_node, token_node, "CONTAINS")
        return name_node

    def add_token(self, token):
        n, added = self.add_node(TokenNode, token=token.lower())
        return n

    def import_instance(self, dataset_name, instance_id, phrase, mappings):
        dataset_node = self.add_dataset(dataset_name=dataset_name)
        instance_node = self.add_instance(instance_id, phrase)
        for terminology, mvalues in mappings.items():
            if not isinstance(mvalues, list):
                mvalues = [mvalues]
            mvalues = Terminology.validate(mvalues, terminology)
            for mcode in mvalues:
                c = self.add_concept(concept_id=mcode,
                                     origin_terminology=terminology,
                                     term_func=get_term_func(terminology),
                                     parents_func=get_parents_func(terminology),
                                     props_func=get_props_func(terminology))
                # self.add_relation(c, term_node, "EXISTS_IN")
                self.add_relation(instance_node, c, "ANNOTATED_AS")
        self.add_relation(dataset_node, instance_node, "HAS_INSTANCE")

    def add_concept_mappings(self, mapper, concept_node):
        has_mapping_to = any(
            (d.get('name', None) == "MAPPED_TO"
             and d.get('mapper', None) == mapper.__class__.__name__)
            for u, v, d in
            self.mdg.edges(concept_node, data=True))
        if has_mapping_to:
            return set()

        all_terminologies = Terminology.all()
        node_term = Terminology(concept_node['origin_terminology'])
        all_terminologies.remove(node_term)
        mapping_nodes = set()
        for term in all_terminologies:
            cands = mapper.map_concept_id(
                source_terminology=node_term,
                target_terminology=term,
                source_concept_id=concept_node['concept_id']
            )
            mapping_nodes.update([self.add_concept(
                concept_id=c,
                origin_terminology=term,
                term_func=get_term_func(term),
                parents_func=get_parents_func(term),
                props_func=get_props_func(term)) for c in cands])
        return mapping_nodes

    def add_mappings(self, mapper):
        concept_nodes = [n for n in self.mdg.nodes()
                         if isinstance(n, OntoConceptNode)]
        return self.add_nodes_mappings(mapper, concept_nodes, recursive=True)

    def add_pt_mappings(self, mapper):
        concept_nodes = [n for n in self.mdg.nodes()
                         if isinstance(n, OntoConceptNode)
                         and n.get('hier_level', None) == 'PT']
        return self.add_nodes_mappings(mapper, concept_nodes)

    def add_nodes_mappings(self, mapper, concept_nodes, recursive=False):
        mapper_name = mapper.__class__.__name__

        recursive_nodes = set()
        for node in tqdm(concept_nodes, desc='process concept nodes'):
            mnodes = self.add_concept_mappings(mapper, node)
            for mnode in mnodes:
                self.add_relation(node, mnode, "MAPPED_TO", mapper=mapper_name)
            if recursive:
                recursive_nodes.update(mnodes)
        if recursive:
            self.add_nodes_mappings(mapper, recursive_nodes, recursive=False)

    def get_node_edges(self, node, edge_name):
        # return [(u, v, d) for u, v, d in self.mdg.edges(node, data=True)
        #         if d.get('name', None) == edge_name]
        return get_node_edges(self.mdg, node, edge_name)

    def get_node_in_edges(self, node, edge_name):
        return get_node_in_edges(self.mdg, node, edge_name)

    def get_concept_mappings(self, concept_node, max_depth=0):
        return get_recursive_nodes(self.mdg, concept_node, "MAPPED_TO",
                                   depth=max_depth)

    def get_concept_parents(self, concept_node, max_depth=0):
        return get_recursive_nodes(self.mdg, concept_node, "IS_A",
                                   depth=max_depth)

    def get_recursive_nodes(self, node, edge_name, depth=0, nodes=None,
                            verbose=0):

        return get_recursive_nodes(self.mdg, node, edge_name, depth=depth,
                                   nodes=nodes, verbose=verbose)

    def get_recursive_nodes_reverse(self, node, edge_name, depth=0, nodes=None):
        return get_recursive_nodes_reverse(self.mdg, node, edge_name,
                                           depth=depth, nodes=nodes)

    def get_concept_closest_meddra_hier(self, concept_node, hier_level='PT',
                                        max_depth=0):
        return get_concept_closest_meddra_hier(self.mdg, concept_node,
                                               hier_level, max_depth)

    def get_instance_meddra_pt(self, instance_node, max_depth=0):
        return get_instance_meddra_pt(self.mdg, instance_node,
                                      max_depth=max_depth)

    def get_annotations_distance(self, a, b):
        if self.udg is None:
            self.preload_dist_graph()

        c1, c2 = min(a['concept_id'], b['concept_id']), max(a['concept_id'],
                                                            b['concept_id'])
        d = self.__ann_distances.get(c1, {}).get(c2, None)

        if d is None:
            d = get_annotations_distance((a, b), g=self.udg)
            self.__ann_distances.setdefault(c1, {})[c2] = d
            # print((a['term'], b['term'], d))
        return d

    def write_relations(self, output_file):

        self.preload_dist_graph(weighted=False,
                                multi=True, concepts_only=False)
        nmap_inv = {str(v): k for k, v in self.udg_nodemap.items()}
        with codecs.open(output_file, 'w', 'utf-8') as fp:
            for u, v, rel in tqdm(self.udg.edges(data=True),
                                  desc='writing lines'):
                h = nmap_inv[str(u)]
                t = nmap_inv[str(v)]
                r = rel['name']
                fp.write("%s\t%s\t__label__%s" % (h, r, t))
                fp.write("\n")
                fp.write("%s\tREVERSE_%s\t__label__%s" % (t, r, h))
                fp.write("\n")

    def build_node2vec_model(self, number_walks=10,
                             walk_length=40, seed=0,
                             dim=50, window_size=5,
                             workers=0, sg=0, hs=0, concepts_only=False):
        from node2vec import Node2Vec
        if workers <= 0:
            workers = multiprocessing.cpu_count()
        self.preload_dist_graph(weighted=True, concepts_only=concepts_only)

        n2v = Node2Vec(self.udg, dimensions=dim, walk_length=walk_length,
                       num_walks=number_walks, workers=workers)
        model = n2v.fit(window=window_size, min_count=1, sg=sg, hs=hs,
                        callbacks=[GensimEpochLogger()])
        return model

    def build_deepwalk_model(self, number_walks=10,
                             walk_length=40,
                             seed=0, representation_size=64,
                             window_size=5,
                             workers=0, epochs=50, sg=0, hs=0,
                             concepts_only=False):
        from deepwalk.graph import load_edgelist, build_deepwalk_corpus
        # self.preload_dist_graph(weighted=False, concepts_only=concepts_only)
        self.preload_merged_dist_graph()
        nmap_inv = {}
        for k, v in self.udg_nodemap.items():
            nmap_inv.setdefault(str(v), []).append(k)
        assert nx.number_connected_components(self.udg) == 1
        # for c in nx.connected_components(self.udg):
        #     c = list(c)
        #     if len(c) < 10:
        #         print(c)
        #         for g in c:
        #             print(nmap_inv[str(g)])
        #     print("----")
        # print('Done.')

        if workers <= 0:
            workers = multiprocessing.cpu_count()
        if workers <= 0:
            workers = multiprocessing.cpu_count()
        with tempfile.NamedTemporaryFile(prefix='graph', suffix='.edgelist') \
                as tf:
            tf.flush()
            temp_name = tf.name
            nx.write_edgelist(self.udg, temp_name, data=False)
            # g.write_edgelist(temp_name, data=False)
            print("%s saved" % temp_name)
            G = load_edgelist(temp_name, undirected=True)
        print("Number of nodes: {}".format(len(G.nodes())))
        num_walks = len(G.nodes()) * number_walks
        print("Number of walks: {}".format(num_walks))

        data_size = num_walks * walk_length
        print("Data size (walks*length): {}".format(data_size))
        print("Walking...")
        walks = build_deepwalk_corpus(G, num_paths=number_walks,
                                      path_length=walk_length,
                                      alpha=0,
                                      rand=random.Random(seed))
        print("Training...")
        model = Word2Vec(walks, size=representation_size,
                         window=window_size, min_count=0, sg=sg, hs=hs,
                         iter=epochs,
                         workers=workers, callbacks=[GensimEpochLogger()])
        keys = list(model.wv.vocab.keys())
        wc = len(self.udg_nodemap)
        awc = 0
        dim = model.vector_size
        if len(nmap_inv) != len(keys):
            kd = set(nmap_inv.keys()).difference(set(keys))
            print("MISSING GROUPS:")
            print(kd)
            for g in kd:
                print(nmap_inv[str(g)])
            raise RuntimeError("Groups are inconsistent!")

        with tempfile.NamedTemporaryFile(mode='w', prefix='gemb',
                                         suffix='.txt') as tf:
            tf.write("%d %d\n" % (wc, dim))
            for w in keys:
                v = model.wv.vocab.pop(w)
                vec = model.wv.vectors[v.index]
                for inv_w in nmap_inv[w]:
                    tf.write("%s %s" % (inv_w, " ".join([str(i) for i in vec])))
                    tf.write("\n")
                    awc += 1
            tf.seek(0)
            tf.flush()
            assert wc == awc
            model = KeyedVectors.load_word2vec_format(tf.name, binary=False)
        return model

    def get_ambiguous_tokens(self, instance_ids=None):
        if self.udg is None:
            self.preload_dist_graph()

        rows = []
        if instance_ids:
            instance_nodes = [node for node in self.mdg.nodes()
                              if isinstance(node, InstanceNode)
                              and node['instance_id'] in instance_ids]

        else:
            instance_nodes = [node for node in self.mdg.nodes()
                              if isinstance(node, InstanceNode)]

        token_map = {}
        token_inst_cnt = {}
        token_phrases = {}
        for node in tqdm(instance_nodes, desc='processing instances'):
            annotations = [v for _, v, _ in self.get_node_edges(node,
                                                                "ANNOTATED_AS")]
            phrase = [v for _, v, _ in self.get_node_edges(node,
                                                           "DESCRIBED_AS")][0]
            tokens = [t.lower() for t in word_tokenize(phrase['phrase'])]
            for t in tokens:
                token_map.setdefault(t, []).extend(annotations)
                token_inst_cnt.setdefault(t, 0)
                token_inst_cnt[t] += 1
                for a in set(annotations):
                    token_phrases.setdefault(t, {}).setdefault(
                        a['concept_id'], []).append((node['instance_id'],
                                                     phrase))

        all_pairs = {}
        for token, t_annotations in token_map.items():
            unq_annotations = set(t_annotations)
            if len(unq_annotations) < 2:
                continue

            # t_counter = Counter(t_annotations)
            # most_popular = t_counter.most_common(1)[0][0]
            # for cand in unq_annotations:
            #     if cand == most_popular:
            #         continue
            #     k = get_concept_pair_key(most_popular, cand)
            #     all_pairs[k] = (most_popular, cand)

            pairs = combinations(unq_annotations, 2)
            for a, b in pairs:
                all_pairs[get_concept_pair_key(a, b)] = (a, b)

        distfunc = partial(get_keyed_emb_annotations_distance,
                           emb=self.graph_embeddings)
        all_pairs = list(all_pairs.values())
        res = [distfunc(p) for p in tqdm(all_pairs,
                                         desc='calculating distances')]
        distmap = dict(res)
        pbar = tqdm(token_map.items(), desc='processing tokens')
        for token, t_annotations in pbar:
            unq_annotations = set(t_annotations)
            if len(unq_annotations) < 2:
                continue
            t_counter = Counter(t_annotations)
            most_popular = t_counter.most_common(1)[0][0]
            # longest_dist = -np.inf
            # print("%s (%d)" % (token, len(unq_annotations)))
            # t_distances = {most_popular: 0}
            # most_distant = None
            # for cand in unq_annotations:
            #     if most_popular == cand:
            #         continue
            #     k = get_concept_pair_key(most_popular, cand)
            #     dist = distmap[k]
            #     if dist > longest_dist:
            #         longest_dist = dist
            #         most_distant = cand
            #     amb_annotations.add(most_popular)
            #     amb_annotations.add(cand)
            #     t_distances[cand] = dist

            t_frequencies = dict(t_counter)
            n_total = sum(t_frequencies.values())
            t_distances = {}
            for a, b in permutations(unq_annotations, 2):
                t_distances.setdefault(a, []).append(
                    distmap[get_concept_pair_key(a, b)] * t_frequencies[b])
            t_distances = {
                k: 1.0 * np.sum(v) / (n_total - 1)
                for k, v in t_distances.items()
            }
            most_distant = max(t_distances.items(), key=operator.itemgetter(
                1))[0]
            longest_dist = t_distances[most_distant]
            dists = np.asarray(list(t_distances.values()))
            ds = (np.sum(dists) - longest_dist) / (len(dists) - 1)
            score = longest_dist - ds

            amb_annotations = list(t_distances.keys())
            t_amb = {a: t_frequencies[a] for a in amb_annotations}
            n_concepts = len(t_amb.keys())
            n_instances = token_inst_cnt[token]
            inst_per_conc = 1.0 * n_instances / n_concepts

            mp_phrase = token_phrases[token][most_popular['concept_id']][0]
            md_phrase = token_phrases[token][most_distant['concept_id']][0]

            mp_phrase = "%s (%s)" % (mp_phrase[1]['phrase'], mp_phrase[0])
            md_phrase = "%s (%s)" % (md_phrase[1]['phrase'], md_phrase[0])
            md_instances = [p[0]
                            for p in
                            token_phrases[token][most_distant['concept_id']]]

            info = "; ".join(["%s (%d) [%.2f]" % (k['term'], v, t_distances[k])
                              for k, v in sorted(t_amb.items(),
                                                 key=lambda x: x[1],
                                                 reverse=True)])
            cids = " ".join([k['node_id'] for k in t_amb.keys()])
            rows.append((
                token,
                most_popular['term'], most_distant['term'],
                mp_phrase, md_phrase,
                longest_dist, score,
                " ".join(md_instances),
                n_instances, n_concepts,
                inst_per_conc, info, cids
            ))
        rows = sorted(rows, key=lambda x: x[6], reverse=True)
        return rows

    def get_annotation_mappings(self, annotations):
        ann_mappings = {}
        for ann in annotations:
            parents = self.get_recursive_nodes(ann, 'IS_A')
            mappings = set()
            for p in parents:
                mappings.update(self.get_recursive_nodes(p, "MAPPED_TO"))
            # ann_mappings[ann['concept_id']] = mappings
            ann_mappings[ann['concept_id']] = []
            for m in mappings:
                ann_mappings[ann['concept_id']].extend(
                    self.get_recursive_nodes(m, "IS_A"))
        return ann_mappings

    def get_unrelated_annotations(self, instance_ids=None, dist_thresh=4):
        rows = []
        mapping_rows = []

        if instance_ids:
            instance_nodes = [node for node in self.mdg.nodes()
                              if isinstance(node, InstanceNode)
                              and node['instance_id'] in instance_ids]

        else:
            instance_nodes = [node for node in self.mdg.nodes()
                              if isinstance(node, InstanceNode)]

        for node in tqdm(instance_nodes, desc='processing instances'):
            annotations = [v for _, v, _ in self.get_node_edges(node,
                                                                "ANNOTATED_AS")]
            phrase = [v for _, v, _ in self.get_node_edges(node,
                                                           "DESCRIBED_AS")][0]

            if len(annotations) < 2:
                continue

            ann_mappings = self.get_annotation_mappings(annotations)

            # TODO: Change to dist calculation between most popular and others
            pairs = combinations(annotations, 2)
            for a, b in pairs:
                dist = get_annotations_distance((a, b), self.mdg)
                if dist > dist_thresh:
                    rows.append((node['instance_id'], phrase['phrase'], dist,
                                 a['concept_id'], a['term'],
                                 b['concept_id'], b['term']))
                    mapping_rows.append((a['concept_id'], a['term'],
                                         b['concept_id'], b['term'],
                                         dist))

        # sort and combine

        rows = sorted(rows, key=lambda x: x[2], reverse=True)
        combined = [list(row) + [frq]
                    for row, frq in Counter(mapping_rows).items()]
        combined = sorted(combined, key=lambda x: x[5] ** x[4], reverse=True)
        return rows, combined

    def generate_dataset(self, instance_ids=None, parallel=True, max_depth=0,
                         update_interval=10):
        rows = {}

        if instance_ids:
            instance_nodes = [node for node in self.mdg.nodes()
                              if isinstance(node, InstanceNode)
                              and node['instance_id'] in instance_ids]

        else:
            instance_nodes = [node for node in self.mdg.nodes()
                              if isinstance(node, InstanceNode)]
        all_annotations = [
            v for node in instance_nodes
            for _, v, _ in self.get_node_edges(node,
                                               "ANNOTATED_AS")

        ]

        ts = time.time()
        if parallel:
            print("Processing nodes in parallel..")
            p = Pool()
            pnf = partial(process_node, mdg=self.mdg,
                          all_annotations=all_annotations, max_depth=max_depth)

            rs = p.map_async(pnf, instance_nodes)
            p.close()
            n_total = len(instance_nodes)
            tick = 1
            n_ticks = 0
            while not rs.ready():
                time.sleep(tick)
                n_ticks += 1
                if tick * n_ticks > update_interval:
                    n_ticks = 0
                    total_left = rs._number_left * rs._chunksize
                    if total_left > n_total:
                        total_done = 0
                    else:
                        total_done = n_total - total_left
                    perc = 100.0 * total_done / n_total
                    print("%.2f%%  (%d/%d) completed" % (
                        perc, total_done, n_total))
            results = rs.get()
            rows = dict(results)
        else:
            for node in tqdm(instance_nodes,
                             desc='Processing instances'):
                node_result = process_node(node, self.mdg, all_annotations,
                                           max_depth=max_depth)
                rows[node_result[0]] = node_result[1]
        print("Done in %.4f sec." % (time.time() - ts))
        return rows

    def save_graph(self, path):
        makedirs_file(path)
        nx.write_gpickle(self.mdg, path)

    def preload_merged_dist_graph(self):
        self.udg = nx.Graph()
        self.udg_nodemap = {}

        node_mappings = {}
        root_nodes = []
        for n in tqdm(self.mdg.nodes(), desc='processing nodes'):
            if not isinstance(n, OntoConceptNode):
                continue
            if n['origin_terminology'] == 'UMLS':
                continue
            mnodes = get_recursive_nodes(self.mdg, n, edge_name='MAPPED_TO')
            mnodes.update(get_recursive_nodes_reverse(self.mdg, n,
                                                      edge_name='MAPPED_TO'))
            node_mappings[n['node_id']] = set([
                m['node_id'] for m in mnodes])
            if not any(r['name'] == 'IS_A'
                       for u, v, r in self.mdg.edges(n, data=True)):
                root_nodes.append(n['node_id'])
        # print(root_nodes)
        root = RootNode("ROOT")
        self.udg_nodemap[root['node_id']] = 0
        for n, mappings in node_mappings.items():
            group_id = None
            if n in self.udg_nodemap:
                group_id = self.udg_nodemap[n]
            for m in mappings:
                if m in self.udg_nodemap:
                    if group_id is None:
                        group_id = self.udg_nodemap[m]
                    assert self.udg_nodemap[m] == group_id
            if group_id is None:
                if len(self.udg_nodemap) > 0:
                    group_id = max(self.udg_nodemap.values()) + 1
                else:
                    group_id = 0

            self.udg_nodemap[n] = group_id
            for m in mappings:
                self.udg_nodemap[m] = group_id

        print("%d nodes grouped into %d groups" % (
            len(self.udg_nodemap), max(self.udg_nodemap.values())))

        for rn in root_nodes:
            self.udg.add_edge(self.udg_nodemap[rn], self.udg_nodemap[root])

        for (u, v, rel) in tqdm(self.mdg.edges(data=True),
                                desc='processing edges'):
            if rel['name'] != 'IS_A':
                continue
            if not isinstance(u, OntoConceptNode):
                continue
            if not isinstance(v, OntoConceptNode):
                continue
            self.udg.add_edge(self.udg_nodemap[u['node_id']],
                              self.udg_nodemap[v['node_id']])

    def preload_dist_graph(self, weighted=False, multi=False,
                           concepts_only=True):
        # self.udg = nx.MultiGraph()
        if multi:
            self.udg = nx.MultiDiGraph()
        else:
            self.udg = nx.Graph()
        self.udg_nodemap = {}
        # nodemap[g.root_node.node_id] = 0

        # 1. get all concept nodes
        # get all names
        # get all names tokens

        concept_nodes = []
        name_nodes = []
        token_nodes = []
        for n in self.mdg.nodes():
            if not isinstance(n, OntoConceptNode):
                continue
            concept_nodes.append(n)
            for (u, v, rel) in self.mdg.edges(n, data=True):
                if rel.get('name', None) == 'NAMED_AS':
                    name_nodes.append(v)
        for n in name_nodes:
            for (u, v, rel) in self.mdg.edges(n, data=True):
                if rel.get('name', None) == 'CONTAINS':
                    token_nodes.append(v)
        if concepts_only:
            print("%d concepts" % (len(concept_nodes)))
            all_nodes = concept_nodes
        else:
            print("%d concepts, %d names, %d tokens" % (len(concept_nodes),
                                                        len(name_nodes),
                                                        len(token_nodes)))

            all_nodes = concept_nodes + name_nodes + token_nodes

        for n in tqdm(all_nodes, desc='processing nodes'):
            node_id = len(self.udg_nodemap.keys())
            if n['node_id'] not in self.udg_nodemap:
                self.udg_nodemap[n['node_id']] = node_id
            self.udg.add_node(node_id)
        if weighted or multi:
            weights = {
                'IS_A': 2.0,
                'MAPPED_TO': 10.0,
                'NAMED_AS': 1.0,
                'CONTAINS': 1.0,
            }

            for (u, v, rel) in self.mdg.edges(data=True):
                if weighted and rel['name'] not in weights:
                    continue
                if u['node_id'] not in self.udg_nodemap or v['node_id'] not \
                        in self.udg_nodemap:
                    continue
                r = {}
                if weighted:
                    r['weight'] = weights[rel['name']]
                if multi:
                    r['key'] = rel['name']
                    r['name'] = rel['name']

                self.udg.add_edge(self.udg_nodemap[u['node_id']],
                                  self.udg_nodemap[v['node_id']], **r)
        else:
            for (u, v) in tqdm(self.mdg.edges(),
                               desc='processing edges'):
                if u['node_id'] not in self.udg_nodemap or v['node_id'] not \
                        in self.udg_nodemap:
                    continue
                self.udg.add_edge(self.udg_nodemap[u['node_id']],
                                  self.udg_nodemap[v['node_id']])

        print("Graph nodes: %d edges: %d" % (self.udg.number_of_nodes(),
                                             self.udg.number_of_edges()))

    def load_graph(self, path):
        self.mdg = nx.read_gpickle(path)
        self.__ann_distances = {}

    def load_embeddings(self, path, binary=True):
        self.graph_embeddings = KeyedVectors.load_word2vec_format(
            path, binary=binary)
