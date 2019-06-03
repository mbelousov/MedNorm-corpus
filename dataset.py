import codecs
import getpass
import json
import multiprocessing
import os
import re
import time
from collections import Counter
from collections import OrderedDict
from itertools import combinations

import Levenshtein
import numpy as np
import pandas as pd
import unidecode
from fire import Fire
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, \
    CountVectorizer, TfidfVectorizer
from tqdm import tqdm

from mednorm.datasets import create_converter
from mednorm.ontology import MedNormOntology, OntoConceptNode
from mednorm.ontology.mapping import BioPortalTerminologyMapper
from mednorm.ontology.mapping import RuleBasedTerminologyMapper
from mednorm.ontology.mapping import Terminology
from mednorm.utils import normpath, load_yaml, makedirs_file, normlist, flatten
from mednorm.utils import print_bold, print_success


def _string_sim_ratio(s1, s2):
    def _norm_term(term):
        if term.endswith(')'):
            return term[:term.rindex('(')].strip()
        return term

    return Levenshtein.ratio(_norm_term(s1), _norm_term(s2))


def _filter_most_similar(emb_model, target_word, vocab_words, topn=300, thresh=None):
    result = []
    topn_ = 1000
    if isinstance(topn, int):
        topn_ = topn * 20

    for w, s in emb_model.most_similar(target_word, topn=topn_):
        if w not in vocab_words:
            continue

        if isinstance(topn, int) and len(result) >= topn:
            result.append((w, s))
            return result
        elif thresh and s < thresh:
            return result
        else:
            result.append((w, s))

    print("Only found %d words (topn=%s, tresh=%s)" % (len(result), topn, thresh))
    return result


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    top_items = sorted_items[:topn] if topn > 0 else sorted_items
    return {feature_names[idx]: sc for idx, sc in top_items}


class TermTokenizer(RegexpTokenizer):
    def __init__(self):
        RegexpTokenizer.__init__(self, r'\s+|[^\-\_A-Za-z0-9]+', gaps=True)


# class TermTokenizer(WordPunctTokenizer):
#     pass
def connect_to_umls_db(*args, **kwargs):
    from pymedtermino.umls import connect_to_umls_db as pymed_connect
    return pymed_connect(*args, **kwargs)


# Pipeline:
# combine -> build graph -> unrelated_annotations -> token_confusions ->
# human_correct -> build_graph -> tsv (only all terminologies) -> resolve

class MedNormDatasetCli(object):
    def combine(self, config, output, sep='\t', encoding='utf-8'):
        """
        Combines all datasets into a single unified corpus
        Parameters
        ----------
        config : Path to the config yaml file
        output : Path to the output file
        sep : Separator (\t or ,)
        encoding : Encoding (utf-8)
        -------

        """
        config_path = normpath(config)
        output_path = normpath(output)

        datasets = load_yaml(config_path)

        # combine datasets
        combined = []
        print_bold("Processing datasets: ")
        for ds_name, ds_path in datasets.items():
            print("\t[*] %s" % ds_name)
            c = create_converter(ds_name, ds_path)
            converted = c.convert_lines()
            combined.extend(converted)
            c.print_stats(padding='\t  ')
            print("\t  %d converted lines" % len(converted))

        # save combined output file
        output_path = normpath(output_path)
        makedirs_file(output_path)
        with codecs.open(output_path, 'w', encoding) as fp:
            fp.write("\t".join(["original_dataset", "instance_id", "phrase",
                                "meddra_code", "sct_id", "umls_cui"]))
            fp.write("\n")
            fp.write("\n".join([sep.join(l) for l in combined]))

        print_success("File %s saved." % output_path)
        print_success("Done. %d lines combined." % len(combined))

    def human_correct(self, dataset, corrections, output):
        dataset_path = normpath(dataset)
        corrections_path = normpath(corrections)
        output_path = normpath(output)

        df = pd.read_csv(dataset_path, sep='\t', dtype='str', encoding='utf-8')

        print("Reading corrections..")
        corr_df = pd.read_csv(corrections_path, sep='\t', dtype='str',
                              encoding='utf-8')
        labels = corr_df.columns.values[1:]
        corrs = dict(zip(corr_df['instance_id'].values,
                         zip(*[corr_df[l].values for l in labels])))
        print("%d corrections provided." % len(corr_df.index))
        skipped = []
        n_corrected = 0
        for idx, row in df.iterrows():
            if row['instance_id'] not in corrs.keys():
                continue
            mapping = corrs[row['instance_id']]

            if all(v is None for v in mapping):
                skipped.append(idx)
                continue

            for i, l in enumerate(labels):
                df.at[idx, l] = mapping[i]
            n_corrected += 1

        df = df.drop(df.index[skipped])
        df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        print("%d instance removed, %d corrected" % (len(skipped), n_corrected))

    def build_graph(self, dataset, output, sep='\t',
                    rules=None,
                    mysql_host='localhost', mysql_user='root', mysql_db="umls"):
        """
        Builds a graph representation for annotated dataset
        Parameters
        ----------
        dataset : Path to the annotated dataset
        output : Graph output path
        sep : Separator
        rules : Path to rules file
        mysql_host : MySQL host
        mysql_user : MySQL user
        mysql_db : MySQL database name

        -------

        """
        dataset_path = normpath(dataset)
        output_path = normpath(output)

        mysql_password = getpass.getpass(prompt='MySQL Password: ')
        connect_to_umls_db(mysql_host, mysql_user, mysql_password,
                           db_name=mysql_db,
                           encoding="latin1")

        df = pd.read_csv(dataset_path, sep=sep, dtype='str', encoding='utf-8')
        mno = MedNormOntology()

        for idx, row in tqdm(df.iterrows(), total=len(df.index),
                             desc='reading dataset'):
            meddra_codes = []
            if 'meddra_code' in row and not pd.isna(row['meddra_code']):
                meddra_codes = row['meddra_code'].split()
            sct_ids = []
            if 'sct_id' in row and not pd.isna(row['sct_id']):
                sct_ids = row['sct_id'].split()

            umls_cuis = []
            if 'umls_cui' in row and not pd.isna(row['umls_cui']):
                umls_cuis = row['umls_cui'].split()

            mappings = {
                Terminology.MEDDRA: meddra_codes,
                Terminology.SNOMEDCT: sct_ids,
                Terminology.UMLS_CUI: umls_cuis,
            }
            mno.import_instance(row['original_dataset'],
                                row['instance_id'],
                                row['phrase'],
                                mappings)

        mapper = BioPortalTerminologyMapper()
        rule_mapper = RuleBasedTerminologyMapper(rules)

        mno.add_mappings(mapper)
        mno.add_mappings(rule_mapper)

        print("PT mappings..")
        mno.add_pt_mappings(mapper)
        mno.add_pt_mappings(rule_mapper)
        mno.save_graph(output_path)

    def resolve_dups(self, dataset, target_col, output, sep='\t',
                     ignore_case=True, keep_ratio=0.0):
        dataset_path = normpath(dataset)
        target_col = normlist(target_col)
        output_path = normpath(output)
        keep_ratio = float(keep_ratio)
        print("Keep ratio: %.2f" % keep_ratio)
        df = pd.read_csv(dataset_path, sep=sep, dtype='str', encoding='utf-8')
        pm = {}
        unq_phrases = set()
        for idx, row in tqdm(df.iterrows(), total=len(df.index)):
            phrase = row['phrase']
            if ignore_case:
                phrase = phrase.lower()
            unq_phrases.add(phrase)
            for tc in target_col:
                pm.setdefault(phrase, {}).setdefault(tc, []).append(row[tc])

        pm = {
            phrase: {
                tc: list(Counter(l).items())
                for tc, l in tcl.items() if len(set(l)) > 1
            }
            for phrase, tcl in pm.items()
        }
        pm = {phrase: tcl for phrase, tcl in pm.items() if len(tcl) > 0}
        print("Conflict phrases: %d (out of %d unique)" % (len(pm),
                                                           len(unq_phrases)))
        n_changed = 0
        for idx, row in tqdm(df.iterrows(), total=len(df.index)):
            phrase = row['phrase']
            if ignore_case:
                phrase = phrase.lower()
            if phrase not in pm:
                continue
            n_changed += 1
            for tc, lc in pm[phrase].items():
                # df.at[idx, tc] = " ".join(
                #     set(flatten([l.split(' ') for l, _ in lc])))
                tot = sum([cnt for _, cnt in lc])
                keep = []
                for l, cnt in lc:
                    ratio = 1.0 * cnt / tot
                    if ratio < keep_ratio or cnt == 1:
                        continue
                    keep.append(l)
                if len(keep) == 0:
                    keep = [l for l, _ in lc]
                df.at[idx, tc] = " ".join(
                    set(flatten([l.split(' ') for l in keep])))

        print("Rows changed: %d (out of %d)" % (n_changed, len(df.index)))
        df.to_csv(output_path, sep='\t', index=False, header=True,
                  encoding='utf-8')
        print("%s saved" % output_path)

    def relations(self, graph, output):
        graph_path = normpath(graph)
        output_path = normpath(output)
        mno = MedNormOntology()
        mno.load_graph(graph_path)
        mno.write_relations(output_path)
        print("Done.")

    def convert_starspace(self, input, output, merge='ignore'):
        input_path = normpath(input)
        output_path = normpath(output)
        vocab = {}
        dim = 0
        with codecs.open(input_path, 'r', 'utf-8') as fp:
            for line in fp:
                line = line.strip()
                p = line.split()
                dim = len(p) - 1
                w = p[0]
                if merge == 'ignore':
                    if w.startswith('__label__') or w.startswith('REVERSE_'):
                        continue
                elif merge == 'average':
                    if w.startswith('__label__'):
                        w = w[len('__label__'):]
                vocab.setdefault(w, []).append(np.asarray(
                    [float(v) for v in p[1:]]))
        vocab = {w: np.mean(vv, axis=0) for w, vv in vocab.items()}

        with codecs.open(output_path, 'w', 'utf-8') as fp:
            fp.write("%d %d\n" % (len(vocab), dim))
            for w, wv in vocab.items():
                fp.write("%s %s\n" % (w, " ".join([str(v) for v in wv])))

    def build_embeddings(self, graph, output, n=10, length=40, seed=0,
                         dim=64, w=5, workers=0, epochs=50, sg=1,
                         hs=1, binary=True, mode='deepwalk', only_concepts=0):
        if int(workers) <= 0:
            workers = multiprocessing.cpu_count()
        concepts_only = bool(only_concepts)
        print("MODE: %s" % mode)
        if sg:
            print("Skip-gram model")
        else:
            print("CBOW model")
        print("Threads: %d" % workers)
        print("Dim: %d" % dim)
        graph_path = normpath(graph)
        output_path = normpath(output)
        mno = MedNormOntology()
        mno.load_graph(graph_path)
        ts = time.time()

        if mode == 'deepwalk':
            emb = mno.build_deepwalk_model(
                number_walks=n, walk_length=length, seed=seed,
                representation_size=dim, window_size=w,
                workers=workers, epochs=epochs, sg=sg, hs=hs,
                concepts_only=concepts_only)
        elif mode == 'node2vec':
            emb = mno.build_node2vec_model(number_walks=n,
                                           walk_length=length, seed=seed,
                                           dim=dim, window_size=w,
                                           workers=workers, sg=sg, hs=hs,
                                           concepts_only=concepts_only)
        else:
            raise ValueError("Invalid mode!")
        print("Graph built in %.2f sec." % (time.time() - ts))
        if binary:
            ext = 'bin'
        else:
            ext = 'w2v'
        output_path = "%s_%dn_%dl_%dw_%ddim.%s" % (
            output_path, n, length, w, dim, ext)
        makedirs_file(output_path)
        emb.wv.save_word2vec_format(output_path, binary=binary)
        print("Model %s saved" % output_path)

    def visual_emb(self, graph, embeddings, output, topn=None, labels=None):
        graph_path = normpath(graph)
        embeddings_path = normpath(embeddings)
        output_path = normpath(output)
        mno = MedNormOntology()
        mno.load_graph(graph_path)
        mno.load_embeddings(embeddings_path)

        print("Words: %d" % len(mno.graph_embeddings.vocab))

        outfiletsv = os.path.join(output_path, 'tensor.tsv')
        outfiletsvmeta = os.path.join(output_path, 'metadata.tsv')
        makedirs_file(outfiletsv, remove=True)
        nodemap = {}
        socmap = {}
        cidfrq = {}
        for node in tqdm(mno.mdg.nodes(), desc='processing nodes'):
            if not isinstance(node, OntoConceptNode):
                continue
            if not node['active']:
                continue
            nodemap[node['node_id']] = node
            frq = len(mno.get_recursive_nodes_reverse(
                node, 'ANNOTATED_AS', depth=1))
            cidfrq[node['node_id']] = frq

            if node['origin_terminology'] != 'MEDDRA':
                meddra_node = None
                for rn in mno.get_recursive_nodes(node, "MAPPED_TO"):
                    if rn['origin_terminology'] == 'MEDDRA':
                        meddra_node = rn
                        break
                if meddra_node is None:
                    for rn in mno.get_recursive_nodes_reverse(
                            node, "MAPPED_TO"):
                        if rn['origin_terminology'] == 'MEDDRA':
                            meddra_node = rn
                            break
            else:
                meddra_node = node

            if meddra_node is None:
                socmap[node['node_id']] = None
                # print("Unable to find meddra node for %s!" % node)
                continue

            if meddra_node['hier_level'] == 'SOC':
                socnodes = [meddra_node]
            else:
                socnodes = []
                for rn in mno.get_recursive_nodes(meddra_node, "IS_A"):
                    if rn['hier_level'] == 'SOC':
                        socnodes.append(rn)
            # socmap[node['node_id']] = '_'.join([i['node_id'] for i in socnodes])
            # socmap[node['node_id']] = min([i['node_id'] for i in socnodes])
            socmap[node['node_id']] = [i['node_id'] for i in socnodes]

        topn_soc = 8 - 1
        soc_counter = Counter([vv for v in socmap.values() if v for vv in v])
        soc_counts = dict(soc_counter.items())
        socmap = {nid: max(socids, key=lambda x: soc_counts.get(x, 0)) if socids else 'OTHER'
                  for nid, socids in socmap.items()}

        topn_soc_values = [x for x, _ in soc_counter.most_common(topn_soc)]
        socmap = {nid: soc if soc in topn_soc_values else 'OTHER' for nid, soc in socmap.items()}

        # print(Counter(socmap.values()).most_common(100))

        words = sorted(cidfrq.items(), key=lambda x: x[1], reverse=True)

        if labels:
            labels_path = normpath(labels)
            with codecs.open(labels_path, 'r', 'utf-8') as fp:
                filter_labels = [l.strip() for l in fp]
            words = [(w, frq) for w, frq in words if w in filter_labels]
        if topn and isinstance(topn, int):
            words = words[:int(topn)]

        words_only = [w for w, _ in words]

        print("Number of items: %d" % len(words))

        topn_clusters = 50 - 1

        # items_per_cluster = 10
        # items_thresh = None

        items_per_cluster = None
        items_thresh = 0.75

        clusters = [{w} for w in [w for w in words_only if w.startswith('MEDDRA_')][:topn_clusters]]
        keep_grow = True
        while keep_grow:
            clusters_nb = {}
            cl_items = []
            for c in clusters:
                cl_items.extend(c)
            print("%d items" % len(cl_items))
            for c in clusters:
                sm = []
                for w in c:
                    sm.extend(_filter_most_similar(mno.graph_embeddings, w, words_only, topn=items_per_cluster,
                                                   thresh=items_thresh))
                smd = {}
                for sw, ss in sm:
                    smd.setdefault(sw, []).append(ss)
                smd = {sw: max(ss) for sw, ss in smd.items()}
                smi_all = sorted(smd.items(), key=lambda x: x[1], reverse=True)
                print(smi_all)

                smi = []
                n_smi = 0
                for sw, ss in smi_all:
                    if sw not in cl_items:
                        n_smi += 1

                    if items_per_cluster and n_smi == items_per_cluster:
                        smi.append((sw, ss))
                        break
                    elif items_thresh and ss < items_thresh:
                        break
                    else:
                        smi.append((sw, ss))

                clusters_nb['||'.join(c)] = smi

            inv_nb = {}
            for ch, nbs in clusters_nb.items():
                for w in ch.split('||'):
                    inv_nb.setdefault(w, set()).add(ch)
                for w, _ in nbs:
                    inv_nb.setdefault(w, set()).add(ch)
            merge_with = {}
            keep_grow = False
            for nb, pws in inv_nb.items():
                if len(pws) > 1:
                    keep_grow = True
                    print("%s shared %s" % (nodemap[nb]['term'], pws))
                pwss = []
                for pwh in pws:
                    pwss.extend(pwh.split("||"))
                for pw in pwss:
                    merge_with.setdefault(pw, set())
                    for pw2 in pwss:
                        if pw == pw2:
                            continue
                        merge_with.setdefault(pw, set()).add(pw2)

            clusters = []
            visited = set()
            n_items = 0
            for pw, mvs in merge_with.items():
                if pw in visited:
                    continue
                c = {pw}
                visited.add(pw)
                mvstack = []
                mvstack.extend([v for v in mvs if v not in visited])
                while mvstack:
                    w = mvstack.pop()
                    visited.add(w)
                    c.add(w)
                    if w in merge_with:
                        mvstack.extend([v for v in merge_with[w] if v not in visited])
                clusters.append(c)
                n_items += len(c)

            if len(clusters) < topn_clusters:
                to_add = topn_clusters - len(clusters)
                to_add_items = [w for w in words_only if w.startswith('MEDDRA_')][n_items:n_items + to_add]
                print(" ADD[%d] %s" % (to_add, [nodemap[c]['term'] for c in to_add_items]))
                clusters.extend([{c} for c in to_add_items])
            for i, c in enumerate(clusters):
                print("-- CLUSTER %d --" % i)
                for v in c:
                    print("  %s" % nodemap[v]['term'])
            print('-' * 80)

        clusters_nb = {}
        cl_items = []
        for c in clusters:
            cl_items.extend(c)
        for c in clusters:
            sm = []
            for w in c:
                sm.extend(_filter_most_similar(mno.graph_embeddings, w, words_only, topn=items_per_cluster,
                                               thresh=items_thresh))
            smd = {}
            for sw, ss in sm:
                smd.setdefault(sw, []).append(ss)
            smd = {sw: max(ss) for sw, ss in smd.items()}
            smi_all = sorted(smd.items(), key=lambda x: x[1], reverse=True)

            smi = []
            n_smi = 0
            for sw, ss in smi_all:
                if sw not in cl_items:
                    n_smi += 1

                if items_per_cluster and n_smi == items_per_cluster:
                    smi.append((sw, ss))
                    break
                elif items_thresh and ss < items_thresh:
                    break
                else:
                    smi.append((sw, ss))

            clusters_nb['||'.join(c)] = smi

        for i, c in enumerate(clusters):
            nbs = clusters_nb['||'.join(c)]
            print("-- CLUSTER %d (%d neighbours) --" % (i, len(nbs)))
            for v in c:
                print("  %s" % nodemap[v]['term'])
            print(nbs)
        print('-' * 80)

        cluster_lbls = {w: ('OTHER', 0.0) for w in words_only}
        for clbl, nbs in clusters_nb.items():
            for v in clbl.split('||'):
                cluster_lbls[v] = (clbl, 1.0)

            for v, s in nbs:
                cluster_lbls[v] = (clbl, s)
        n_marked = sum([int(v != 'OTHER') for v, _ in cluster_lbls.values()])
        # print(n_marked)
        # print((topn_clusters * items_per_cluster) + n_items)
        # assert n_marked == (topn_clusters * items_per_cluster) + n_items

        embeddings_vectors = []
        soc_mapped = []
        cluster_lbl_mapped = []
        with open(outfiletsv, 'w+') as file_vector:
            with open(outfiletsvmeta, 'w+') as file_metadata:
                header_row = "\t".join(['term', 'soc', 'cid', 'frq', 'cluster', 'cluster_score'])
                file_metadata.write(header_row + '\n')
                for word, frq in tqdm(words,
                                      desc='processing words'):
                    term = nodemap.get(word, {}).get('term', 'UNK')
                    soc = socmap[word]
                    if soc != 'OTHER':
                        soc = nodemap.get(soc, {}).get('term', 'UNK')

                    soc_mapped.append(soc)
                    cluster_lbl, cluster_score = cluster_lbls[word]
                    if cluster_lbl != 'OTHER':
                        cluster_lbl = '||'.join([nodemap[v]['term'] for v in cluster_lbl.split('||')])
                    cluster_lbl_mapped.append(cluster_lbl)
                    item_row = "\t".join([term, soc, word, str(frq), cluster_lbl, '%.4f' % cluster_score])
                    file_metadata.write(item_row + '\n')
                    embeddings_vectors.append(mno.graph_embeddings[word])
                    vector_row = '\t'.join(
                        str(x) for x in mno.graph_embeddings[word])
                    file_vector.write(vector_row + '\n')
        print("2D tensor file saved to %s" % outfiletsv)
        print("Tensor metadata file saved to %s" % outfiletsvmeta)
        from tensorflow.contrib.tensorboard.plugins import projector
        import tensorflow as tf
        config = projector.ProjectorConfig()
        # embeddings_vectors = mno.graph_embeddings.vectors

        print("Vectors: %d" % len(embeddings_vectors))
        print("Words: %d" % len(words))
        print("SOC values: %d" % len(set(soc_mapped)))
        print(Counter(soc_mapped).most_common(25))
        print("CLUSTER values: %d" % len(set(cluster_lbl_mapped)))
        print(Counter(cluster_lbl_mapped).most_common(25))
        emb = tf.Variable(np.asarray(embeddings_vectors),
                          name='graph_embeddings')
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init_op)

            # Save the variables to disk.
            save_path = saver.save(sess, os.path.join(output_path, 'model.ckpt'))
            print("Model saved in path: %s" % save_path)
        print('Run `tensorboard --logdir={0}` to  visualize result on tensorboard'.format(output_path))

    def ambiguous_tokens(self, dataset, graph, embeddings, output):
        dataset_path = normpath(dataset)
        graph_path = normpath(graph)
        embeddings_path = normpath(embeddings)
        output_path = normpath(output)

        df = pd.read_csv(dataset_path, sep='\t', dtype='str', encoding='utf-8')
        mno = MedNormOntology()
        mno.load_graph(graph_path)
        mno.load_embeddings(embeddings_path)
        instance_ids = set(df['instance_id'].values)
        rows = mno.get_ambiguous_tokens(instance_ids=instance_ids)
        df = pd.DataFrame(rows, columns=[
            'token',
            'most_popular', 'most_distant',
            'most_popular_example', 'most_distant_example',
            'longest_distance', 'score',
            'distant_instances',
            'n_instances', 'n_concepts',
            'inst_per_conc', 'info', "cids"
        ])
        df = df.sort_values(['score'], ascending=False)
        print("Rows: %d" % len(df.index))
        makedirs_file(output_path)
        df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        print("%s saved" % output_path)

    def unrelated_annotations(self, graph, output, dthresh=2):
        graph_path = normpath(graph)
        output_path = normpath(output)

        mno = MedNormOntology()
        mno.load_graph(graph_path)

        rows, combined = mno.get_unrelated_annotations(dist_thresh=dthresh)
        df = pd.DataFrame(rows, columns=[
            'instance_id', 'phrase', 'distance', 'concept_a', 'term_a',
            'concept_b',
            'term_b'
        ])
        print("Rows: %d" % len(df.index))

        mdf = pd.DataFrame(combined, columns=['concept_a', 'term_a',
                                              'concept_b', 'term_b',
                                              'dist', 'frequency'])
        print("Mappings: %d" % len(mdf.index))

        dist_output_path = '%s_dist_combined.tsv' % output_path
        makedirs_file(dist_output_path)
        mdf.to_csv(dist_output_path, sep='\t', index=False, encoding='utf-8')
        filtered_output_path = '%s_distances.tsv' % output_path
        df.to_csv(filtered_output_path, sep='\t', index=False, encoding='utf-8')
        print("%s saved" % output_path)

    def tsv(self, graph, dataset, output, max_depth=0, parallel=False,
            non_empty=False):
        graph_path = normpath(graph)
        dataset_path = normpath(dataset)
        output_path = normpath(output)
        non_empty = bool(non_empty)

        df = pd.read_csv(dataset_path, sep='\t', dtype='str', encoding='utf-8')
        mno = MedNormOntology()
        mno.load_graph(graph_path)
        instance_ids = set(df['instance_id'].values)
        rows = mno.generate_dataset(instance_ids=instance_ids,
                                    parallel=parallel,
                                    max_depth=max_depth)
        df['mapped_meddra_codes'] = None
        df['mapped_sct_ids'] = None

        for idx, row in df.iterrows():
            df.at[idx, 'mapped_meddra_codes'] = " ".join(
                rows[row['instance_id']]['meddra_codes'])
            df.at[idx, 'mapped_sct_ids'] = " ".join(
                rows[row['instance_id']]['sct_ids'])
        if non_empty:
            print("Initial: %d" % len(df.index))
            term_col_names = ['mapped_meddra_codes', 'mapped_sct_ids']
            # for col_name in term_col_names:
            #     df = df[~df[col_name].isnan()]

            df.replace(to_replace={k: '' for k in term_col_names},
                       value=np.nan, inplace=True)
            df.dropna(subset=term_col_names, inplace=True)
            print("After terminology filtering: %d" % len(df.index))

        print("Rows: %d" % len(df.index))
        df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        print("%s saved" % output_path)

    def non_empty(self, dataset, output):
        dataset_path = normpath(dataset)
        output_path = normpath(output)

        df = pd.read_csv(dataset_path, sep='\t', dtype='str', encoding='utf-8')
        print("Initial: %d" % len(df.index))
        term_col_names = ['mapped_meddra_codes', 'mapped_sct_ids']
        df.dropna(subset=term_col_names, inplace=True)
        # for col_name in term_col_names:
        #     df = df[~df[col_name].isnull()]
        print("After terminology filtering: %d" % len(df.index))
        print("Rows: %d" % len(df.index))
        df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        print("%s saved" % output_path)

    def reduce(self, dataset, output, label_col, embeddings, graph, source_col, ignore_case=True):
        dataset_path = normpath(dataset)
        output_path = normpath(output)
        label_col = normlist(label_col)
        embeddings_path = normpath(embeddings)
        graph_path = normpath(graph)

        source_col = normlist(source_col)

        df = pd.read_csv(dataset_path, sep='\t', dtype='str', encoding='utf-8')
        print("Instances: %d" % len(df.index))

        print("loading graph..")
        mno = MedNormOntology()
        mno.load_graph(graph_path)

        emb = KeyedVectors.load_word2vec_format(embeddings_path,
                                                binary=True)
        km = {k.rsplit('_', 1)[1]: k
              for k in emb.vocab.keys()}

        frqs = {}
        for lc in label_col:
            vs = [v.split(' ') if str(v) != 'nan' else []
                  for v in df[lc].values]
            frqs.update(Counter(flatten(vs)))
        source_frqs = {}
        for lc in source_col:
            vs = [v.split(' ') if str(v) != 'nan' else []
                  for v in df[lc].values]
            source_frqs.update(Counter(flatten(vs)))
        base_col = label_col[0]
        print("Base column: %s" % base_col)
        bests = {}
        pm = {}
        source_pm = {}
        unq_phrases = set()
        concepts = set()
        for idx, row in tqdm(df.iterrows(), total=len(df.index)):
            phrase = row['phrase']
            phrase = phrase.lower()
            unq_phrases.add(phrase)
            for tc in label_col:
                pm.setdefault(phrase, {}).setdefault(tc, []).extend(row[tc].split())
                concepts.update(row[tc].split())
            for tc in source_col:
                if pd.isna(row[tc]):
                    continue
                source_pm.setdefault(phrase, {}).setdefault(tc, []).extend(row[tc].split())
                concepts.update(row[tc].split())
        #
        # pm = {
        #     phrase: {
        #         tc: Counter(l)
        #         for tc, l in tcl.items()
        #     }
        #     for phrase, tcl in pm.items()
        # }
        # source_pm = {
        #     phrase: {
        #         tc: Counter(l)
        #         for tc, l in tcl.items()
        #     }
        #     for phrase, tcl in source_pm.items()
        # }
        print("Unique concepts: ", len(concepts))
        nodes = mno.match_concept_nodes(concepts)
        concept_names = {}
        for n in nodes:
            concept_names[n['concept_id']] = n['term']

        print("Matched nodes: ", len(concept_names))

        def _resolve_to_most_common(cands, frqs, source_frqs=None, multi=False):
            if source_frqs:
                best_source_frq = max([source_frqs.get(x, 0) for x in cands])
                best_frq = max([frqs.get(x, 0) for x in cands
                                if source_frqs.get(x, 0) == best_source_frq])
                best_cands = [x for x in cands
                              if frqs.get(x, 0) == best_frq and source_frqs.get(x, 0) == best_source_frq]

            else:
                best_frq = max([frqs.get(x, 0) for x in cands])
                best_cands = [x for x in cands if frqs.get(x, 0) == best_frq]
            if len(best_cands) > 1:
                if multi:
                    return sorted(best_cands)
                # print("pick from multiple: ", best_cands)
                best_base = sorted(best_cands)[0]
                # print("multiple %s" % row['phrase'])
                # print(source_pm[phrase].get(source_col[0], {}))
                # for x in vs:
                #     if source_col:
                #         print("  ", x, source_frqs[source_col[0]][x], frqs[base_col][x])
                #     else:
                #         print("  ", x, frqs[base_col][x])
                # print(best_base)
                # print("------------------------")
            else:
                best_base = best_cands[0]
            if multi:
                return [best_base]
            return best_base

        def _resolve_to_most_similar(candidates, best_base, concept_names, frqs, emb, source_frqs=None, multi=False):
            sims = [emb.similarity(km[x], km[best_base]) for x in candidates]
            best_sim = max(sims)
            cands = [v for v, s in zip(candidates, sims) if s == best_sim]
            if len(cands) > 1:
                s_sims = [
                    _string_sim_ratio(concept_names.get(best_base, best_base), concept_names.get(x, x))
                    for x in cands]
                best_s_sim = max(s_sims)
                best_cands = [v for v, s in zip(cands, s_sims) if s == best_s_sim]
                if len(best_cands) > 1:
                    ls = _resolve_to_most_common(best_cands, frqs, source_frqs)
                else:
                    ls = best_cands[0]
            else:
                ls = cands[0]
            if multi:
                return [ls]
            return ls

        def _resolve_source_similar(slbls, lbls, emb, concept_names, frqs, source_frqs):
            sims = [np.sum([
                (1.0 if x == sl else 0.9 * emb.similarity(km[x], km[sl])) if sl in km else 0.0
                for sl in slbls])
                for x in lbls]
            best_sim = max(sims)
            candidates = [v for v, s in zip(lbls, sims) if s == best_sim]
            if len(candidates) > 1:
                s_sims = [np.sum([
                    _string_sim_ratio(concept_names.get(x, x), concept_names.get(sl, sl))
                    for sl in slbls
                ]) for x in candidates]
                best_s_sim = max(s_sims)
                best_cands = [v for v, s in zip(candidates, s_sims) if s == best_s_sim]
                # print(best_s_sim, best_cands)
                if len(best_cands) > 1:
                    ls = _resolve_to_most_common(best_cands, frqs, source_frqs)
                else:
                    ls = best_cands[0]
            else:
                ls = candidates[0]
            return ls

        # for idx, row in df.iterrows():
        #     # phrase = row['phrase'].lower()
        #     m = {}
        #     for l in label_col:
        #         m[l] = row[l].split(' ') if str(row[l]) != 'nan' else []
        #     if any(len(v) > 1 for v in m.values()):
        #         # get the most popular from the first column
        #         vs = m[base_col]
        #         cands = vs
        #
        #         # best_source_pm_frq = max([source_pm[phrase].get(source_col[0], {}).get(x, 0) for x in vs])
        #         # if best_source_pm_frq > 0:
        #         #     cands = [x for x in vs
        #         #                   if source_pm[phrase].get(source_col[0], {}).get(x, 0) == best_source_pm_frq]
        #         best_bases = _resolve_to_most_common(cands, frqs, source_frqs=source_frqs, multi=True)
        #         if len(best_bases) > 1:
        #             if source_col:
        #                 sources = flatten([row[tc].split(' ') if not pd.isna(row[tc]) else [] for tc in source_col])
        #                 best_base = _resolve_source_similar(sources, best_bases, emb, concept_names, frqs, source_frqs)
        #             else:
        #                 best_base = best_bases[0]
        #         else:
        #             best_base = best_bases[0]
        #
        #         # if len(set([pm[phrase][base_col][x] for x in vs])) > 1:
        #         #     print("--- different values ---")
        #         #     for x in vs:
        #         #         print("  ", x, pm[phrase][base_col][x], frqs[base_col][x])
        #         #     print("------------1------------")
        #         # if len(set([frqs[base_col][x] for x in vs])) > 1:
        #         #     print("--- different frq values %s ---" % row['instance_id'])
        #         #     for x in vs:
        #         #         print("  ", x, pm[phrase][base_col][x], frqs[base_col][x])
        #         #     print("------------------------")
        #
        #         bests.setdefault(base_col, []).append(best_base)
        #         for l in label_col[1:]:
        #             ls = _resolve_to_most_similar(m[l], best_base, concept_names, frqs, emb, source_frqs=source_frqs)
        #             bests.setdefault(l, []).append(ls)
        #     else:
        #         for l, ls in m.items():
        #             bests.setdefault(l, []).append(ls[0])

        # for lc, best_mvs in bests.items():
        reduced = {}
        for idx, row in df.iterrows():
            # debug = row['instance_id'] in ('TwADR-L.fold-0.train.txt_2643', )

            m = {}
            for l in label_col:
                m[l] = row[l].split(' ') if str(row[l]) != 'nan' else []
            slbls = []
            for sc in source_col:
                slbls.extend(row[sc].split(' ') if str(row[sc]) != 'nan' else [])

            base_label = _resolve_source_similar(slbls, m[base_col], emb, concept_names, frqs, source_frqs)
            reduced.setdefault(base_col, []).append(base_label)
            # if debug:
            #     print("------ %s: %s ------" % (row['instance_id'], row['phrase']))
            #     print("Base: ", base_label)
            for l in label_col[1:]:
                ls = _resolve_to_most_similar(m[l], base_label, concept_names, frqs, emb, source_frqs=source_frqs)
                # if debug:
                #     print("  most similar: ", ls)
                reduced.setdefault(l, []).append(ls)
            # if debug:
            #     print("-------------------")

        for lc, best_mvs in reduced.items():
            vs = [v.split(' ') if str(v) != 'nan' else []
                  for v in df[lc].values]
            n_orig_lbl = len(set(flatten(vs)))
            n_reduced = len(set(best_mvs))
            df['single_%s' % lc] = best_mvs
            print("%s\torig: %d reduced: %d single" % (lc, n_orig_lbl,
                                                       n_reduced))

        # for lc in label_col:
        #     vs = [v.split(' ') if str(v) != 'nan' else []
        #           for v in df[lc].values]
        #     n_orig_lbl = len(set(flatten(vs)))
        #     ms_mvs = []
        #     sv = [[v for v in vals if str(v) != 'nan']
        #           for vals in zip(*[df[sc].values for sc in source_col])]
        #     ms_mvs = []
        #     for lbls, slbls in zip(vs, sv):
        #         if len(lbls) == 0:
        #             ms_mvs.append("")
        #             print("SKIP!")
        #             continue
        #         ls = _resolve_source_similar(slbls, lbls, emb, concept_names, frqs, source_frqs)
        #
        #         ms_mvs.append(ls)
        #
        #     # n_best_reduced = len(set(best_mvs))
        #     n_ms_reduced = len(set(ms_mvs))
        #     # df['best_%s' % lc] = best_mvs
        #     if ms_mvs:
        #         df['single_%s' % lc] = ms_mvs
        #     # print("%s\torig: %d reduced: %d best; %d ms" % (lc, n_orig_lbl,
        #     #                                                 n_best_reduced,
        #     #                                                 n_ms_reduced))
        #     print("%s\torig: %d reduced: %d single" % (lc, n_orig_lbl,
        #                                                n_ms_reduced))

        # for lc in label_col:
        #     vs = [v.split(' ') if str(v) != 'nan' else []
        #           for v in df[lc].values]
        #     n_orig_lbl = len(set(flatten(vs)))
        #     ms_mvs = []
        #     if source_col:
        #         sv = [[v for v in vals if str(v) != 'nan']
        #               for vals in zip(*[df[sc].values for sc in source_col])]
        #
        #         ms_mvs = [
        #             max(lbls, key=lambda x: np.sum([
        #                 emb.similarity(km[x], km[sl]) if sl in km else 0.0
        #                 for sl in slbls])) if len(lbls) > 0 else ""
        #             for lbls, slbls in zip(vs, sv)
        #         ]
        #
        #     frq = Counter(flatten(vs))
        #     # mc_inm = {"_".join(sorted(lbls)): (max(lbls, key=lambda x: frq[x])
        #     #                                    if len(lbls) > 0 else "")
        #     #           for lbls in vs}
        #     m_frq = dict(Counter([
        #         max(lbls, key=lambda x: frq[x]) if len(lbls) > 0 else ""
        #         for lbls in vs]).items())
        #     mc_mvs = [
        #         (min(lbls, key=lambda x: m_frq.get(x, np.inf))
        #          if len(lbls) > 0 else "")
        #         for lbls in vs
        #     ]
        #
        #     n_mc_reduced = len(set(mc_mvs))
        #     n_ms_reduced = len(set(ms_mvs))
        #     df['mc_%s' % lc] = mc_mvs
        #     if ms_mvs:
        #         df['ms_%s' % lc] = ms_mvs
        #
        #     print("%s\torig: %d reduced: %d mc; %d ms" % (lc, n_orig_lbl,
        #                                                   n_mc_reduced,
        #                                                   n_ms_reduced))

        target_col = ['single_%s' % lc for lc in label_col]
        pm = {}
        unq_phrases = set()
        for idx, row in tqdm(df.iterrows(), total=len(df.index)):
            phrase = row['phrase']
            if ignore_case:
                phrase = phrase.lower()
            unq_phrases.add(phrase)
            for tc in target_col:
                pm.setdefault(phrase, {}).setdefault(tc, []).append(row[tc])

        pm = {
            phrase: {
                tc: list(Counter(l).items())
                for tc, l in tcl.items() if len(set(l)) > 1
            }
            for phrase, tcl in pm.items()
        }
        pm = {phrase: tcl for phrase, tcl in pm.items() if len(tcl) > 0}
        print("Conflict phrases: %d (out of %d unique)" % (len(pm),
                                                           len(unq_phrases)))

        single_base_col = 'single_%s' % base_col
        n_changed = 0
        for idx, row in tqdm(df.iterrows(), total=len(df.index)):
            phrase = row['phrase']
            if ignore_case:
                phrase = phrase.lower()
            if phrase not in pm:
                continue
            n_changed += 1
            base_lc = pm[phrase].get(single_base_col, [])
            best_cnt = max([cnt for _, cnt in base_lc])
            cands = [l for l, cnt in base_lc if cnt == best_cnt]
            if len(cands) > 1:
                best_base = _resolve_to_most_common(cands, frqs, source_frqs)
            else:
                best_base = cands[0]


            df.at[idx, single_base_col] = best_base

            for tc, lc in pm[phrase].items():
                if tc == single_base_col:
                    continue

                # df.at[idx, tc] = " ".join(
                #     set(flatten([l.split(' ') for l, _ in lc])))
                # best_cnt = max([cnt for _, cnt in lc])
                # cands = [l for l, cnt in lc if cnt == best_cnt]
                # if len(cands) > 1:
                #     keep = _resolve_to_most_common(cands, frqs, source_frqs)
                # else:
                #     keep = cands[0]
                keep = _resolve_to_most_similar([l for l, _ in lc], best_base, concept_names, frqs, emb, source_frqs)
                df.at[idx, tc] = keep
        # quick validation:
        cs = {}
        pm = {}
        for idx, row in df.iterrows():
            phrase = row['phrase'].strip().lower()
            for tc in target_col:
                pm.setdefault(tc, {}).setdefault(phrase, []).append(row[tc])
                cs.setdefault(tc, set()).add(row[tc])

        for tc, tci in pm.items():
            tcm = {p: list(Counter(v).items()) for p, v in tci.items()}
            tcm = {p: v for p, v in tcm.items() if len(v) > 1}
            print("COL: %s, %d duplicates, %d labels" % (tc, len(tcm), len(cs[tc])))
            for phrase, items in tcm.items():
                print("  ", phrase)
                print("    ", items)
        df.to_csv(output_path, sep='\t', header=True, index=False)
        print("%s saved" % output_path)

    def hier(self, dataset, graph, output, label_col):
        dataset_path = normpath(dataset)
        graph_path = normpath(graph)
        output_path = normpath(output)
        label_col = normlist(label_col)
        print("loading graph..")
        mno = MedNormOntology()
        mno.load_graph(graph_path)
        print("reading dataset..")
        df = pd.read_csv(dataset_path, sep='\t', dtype='str', encoding='utf-8')
        print("Instances: %d" % len(df.index))
        hiers = ['PT', 'HLT', 'HLGT', 'SOC']
        hier_map = {}
        for lc in label_col:
            vs = set(df[lc].values)
            for v in tqdm(vs, desc="mapping %s" % lc):
                nd = mno.find_node(OntoConceptNode, concept_id=v)
                if nd is None:
                    raise ValueError("Unable to find node for %s" % v)
                for hl in hiers:
                    mnodes = mno.get_concept_closest_meddra_hier(
                        nd, hier_level=hl)
                    hier_map.setdefault(v, {}).setdefault(hl, set()).update(
                        n['concept_id'] for n in mnodes
                    )
            for hl in hiers:
                mvs = [list(hier_map.get(v, {}).get(hl, set()))
                       for v in list(df[lc].values)]
                n_unq = len(set(flatten(mvs)))

                mcn = "%s_%s" % (lc, hl.lower())
                mvs = [" ".join(v) for v in mvs]
                df[mcn] = mvs
                n_unq_mappings = len(set(mvs))
                print("%s: %d unique labels; %d unique mapping" % (
                    mcn, n_unq, n_unq_mappings))

        df.to_csv(output_path, sep='\t', header=True, index=False)
        print("%s saved" % output_path)

    def filter(self, dataset, output, original, header=1):
        header = bool(header)
        dataset_path = normpath(dataset)
        output_path = normpath(output)
        if isinstance(original, str):
            original = original.split(',')
        df = pd.read_csv(dataset_path, sep='\t', dtype='str')
        print("Initial: %d" % len(df.index))
        df = df[df['original_dataset'].isin(original)]
        print("After original filtering: %d" % len(df.index))
        # rows = []
        # for idx, row in df.iterrows():
        #     inst_id = '%s_%s' % (
        #     row['original_dataset'], row['instance_id'])
        #     rows.append((
        #         inst_id, row['phrase'], row[col_name]
        #     ))
        # result = pd.DataFrame(rows,
        #                       columns=['instance_id', 'phrase', 'labels'])
        df.to_csv(output_path, sep='\t', header=header, index=False)
        print("%s saved" % output_path)

    def minfreq(self, dataset, output, n, target_col, unique=True):
        dataset_path = normpath(dataset)
        output_path = normpath(output)
        unique = bool(unique)

        n = int(n)
        df = pd.read_csv(dataset_path, sep='\t', dtype='str', encoding='utf-8')
        print("Initial: %d" % len(df.index))
        df = df[~df[target_col].isnull()]
        print("After original filtering: %d" % len(df.index))
        if unique:
            df = df.drop_duplicates(subset=['phrase'])
            print("After duplicates filtering: %d" % len(df.index))
        counter = Counter(list(df[target_col].values))
        flt_labels = set([l for l, frq in counter.items() if frq >= n])
        print("Labels: %d (out of %d)" % (len(flt_labels), len(counter.keys())))
        flt_df = df[df[target_col].isin(flt_labels)]
        print("After filtering: %d" % len(flt_df.index))
        if len(flt_df.index) == 0:
            print("Filter returns zero rows!")
            exit(2)

        flt_df.to_csv(output_path, sep='\t', header=True, index=False)
        print("%s saved" % output_path)

    def most_popular(self, dataset, output, n, target_col, unique=True):
        dataset_path = normpath(dataset)
        output_path = normpath(output)
        unique = bool(unique)

        n = int(n)
        df = pd.read_csv(dataset_path, sep='\t', dtype='str', encoding='utf-8')
        print("Initial: %d" % len(df.index))
        df = df[~df[target_col].isnull()]
        print("After original filtering: %d" % len(df.index))
        if unique:
            df = df.drop_duplicates(subset=['phrase'])
            print("After duplicates filtering: %d" % len(df.index))
        counter = Counter(list(df[target_col].values))
        mc = counter.most_common(n)
        flt_labels = set([l for l, frq in mc])
        print("Labels: %d (out of %d)" % (len(flt_labels), len(counter.keys())))
        flt_df = df[df[target_col].isin(flt_labels)]
        print("After filtering: %d" % len(flt_df.index))
        if len(flt_df.index) == 0:
            print("Filter returns zero rows!")
            exit(2)

        flt_df.to_csv(output_path, sep='\t', header=True, index=False)
        print("%s saved" % output_path)



if __name__ == '__main__':
    Fire(MedNormDatasetCli)
