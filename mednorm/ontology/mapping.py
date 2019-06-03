import hashlib
import json
import os
import warnings

from mednorm.pmt_helpers import import_pymedtermino

pymedtermino = import_pymedtermino()

import requests
from enum import Enum
from mednorm.utils import makedirs_file

try:
    from pymedtermino.meddra import MEDDRA
except:
    MEDDRA = None
try:
    from pymedtermino.snomedct import SNOMEDCT
except:
    SNOMEDCT = None
try:
    from pymedtermino.umls import UMLS_CUI
except:
    UMLS_CUI = None

import codecs
from six.moves.urllib.parse import quote_plus

REST_URL = "http://data.bioontology.org"
API_KEY = '846740ef-e76e-4619-9588-c3d9789b40dc'

MAPPING_RULES = {
    'C0043096': 'C1262477',
}


class Terminology(Enum):
    MEDDRA = 'MEDDRA'
    SNOMEDCT = 'SNOMEDCT'
    UMLS_CUI = 'UMLS'

    @classmethod
    def all(cls):
        return list(cls)

    @classmethod
    def get_concept(cls, concept_id, terminology):
        if terminology == Terminology.MEDDRA:
            return MEDDRA.get_by_meddra_code(concept_id)[0]
        elif terminology == Terminology.SNOMEDCT:
            return SNOMEDCT[concept_id]
        elif terminology == Terminology.UMLS_CUI:
            return UMLS_CUI[concept_id]
        raise ValueError("Unknown terminology %s" % terminology)

    @classmethod
    def get_term(cls, concept_id, terminology):
        # if terminology == Terminology.UMLS_CUI:
        #     return "UMLS_%s" % concept_id
        try:
            return cls.get_concept(concept_id, terminology).term
        except ValueError:
            return "%s_%s" % (terminology, concept_id)

    @classmethod
    def get_parents(cls, concept_id, terminology):
        if terminology == Terminology.UMLS_CUI:
            return []

        parents = cls.get_concept(concept_id, terminology).parents
        if terminology == Terminology.MEDDRA:
            return [p.meddra_code for p in parents]
        return [p.code for p in parents]

    @classmethod
    def is_valid(cls, concept_id, terminology):
        try:
            c = Terminology.get_concept(concept_id, terminology)
            if terminology == Terminology.SNOMEDCT:
                parents = c.ancestors_no_double()
                cf = SNOMEDCT['404684003']
                if not any(p == cf for p in parents):
                    print("%s is invalid in %s" % (c, terminology))
                    return False
            return True
        except Exception:
            print("Concept %s is invalid in %s" % (concept_id, terminology))
            return False

    @classmethod
    def validate(cls, concept_ids, terminology):
        valid = [MAPPING_RULES.get(c, c)
                 for c in concept_ids]
        # valid  = concept_ids

        valid = [c for c in valid
                 if cls.is_valid(c, terminology)]
        return valid


class TerminologyMapper(object):
    def map_concept_id(self, source_terminology,
                       target_terminology, source_concept_id):
        raise NotImplementedError("not implemented")


class RuleBasedTerminologyMapper(TerminologyMapper):
    def __init__(self, rules_file):
        self._rules_file = rules_file
        self._rules = {}
        self._read_rules()

    def _read_rules(self):
        self._rules = {}
        n_rules = 0
        if self._rules_file:
            print("\n")
            print('{:-^80}'.format(' RULES '))
            with codecs.open(self._rules_file, 'r', 'utf-8') as fp:
                for line in fp:
                    line = line.strip()
                    parts = line.split('\t')
                    if len(parts) < 2:
                        parts = line.split()
                    if len(parts) != 4:
                        continue
                    st, scid, tt, tcid = parts
                    print("%15s %15s \t -> %15s %15s" % (st, scid, tt, tcid))
                    st, tt = st.lower(), tt.lower()

                    self._rules.setdefault(st, {}).setdefault(
                        scid, {}).setdefault(tt, set()).add(tcid)

                    self._rules.setdefault(tt, {}).setdefault(
                        tcid, {}).setdefault(st, set()).add(scid)

                    n_rules += 1
        print('-' * 80)
        print("%d rules added" % n_rules)

    def map_concept_id(self, source_terminology,
                       target_terminology, source_concept_id):
        st = source_terminology.value.lower()
        tt = target_terminology.value.lower()
        candidates = self._rules.get(st, {}).get(
            source_concept_id, {}).get(tt, set())
        if self._rules.get(st, {}).get(
                source_concept_id, {}):
            print("Rule matched for %s" % source_concept_id)
        candidates = Terminology.validate(candidates, target_terminology)
        return candidates


class BioPortalTerminologyMapper(TerminologyMapper):
    def __init__(self):
        self._cache = {}

    def add_cached(self, source_terminology, target_terminology,
                   source_concept_id,
                   target_concept_ids):
        if not Terminology.is_valid(source_concept_id, source_terminology):
            print("%s is invalid concept" % source_concept_id)
            return
        target_concept_ids = Terminology.validate(target_concept_ids,
                                                  target_terminology)
        self._cache.setdefault(source_terminology, {}).setdefault(
            target_terminology, {})[str(source_concept_id)] = set(
            [str(t) for t in target_concept_ids])

    def add_cached_from_file(self, source_terminology, target_terminology,
                             input_file):
        with open(input_file, 'r') as fp:
            for line in fp:
                parts = line.strip().split("\t")
                source_concept_id = parts[0]
                target_concept_ids = set()
                if len(parts) > 1:
                    target_concept_ids = set(parts[1].split())
                self.add_cached(source_terminology, target_terminology,
                                source_concept_id, target_concept_ids)

    def get_cached(self, source_terminology, target_terminology,
                   source_concept_id):
        return self._cache.get(source_terminology, {}).get(
            target_terminology, {}).get(str(source_concept_id), None)

    def _get_cached_url_fpath(self, url):
        return os.path.join('.cache/', hashlib.md5(url.encode(
            'utf-8')).hexdigest())

    def _get_cached_url(self, url):
        fpath = self._get_cached_url_fpath(url)
        try:
            with codecs.open(fpath, 'r', 'utf-8') as fp:
                return fp.read()
        except IOError:
            return None

    def _set_cached_url(self, url, content):
        fpath = self._get_cached_url_fpath(url)
        makedirs_file(fpath)
        with codecs.open(fpath, 'w', 'utf-8') as fp:
            fp.write(content)

    def _get_json(self, url):
        content = self._get_cached_url(url)
        if content is None:
            # print("fetch %s " % url)
            r = requests.get(url,
                             headers={
                                 'Authorization': 'apikey token=%s' % API_KEY})
            content = r.text
            self._set_cached_url(url, content)
        return json.loads(content)

    def _get_resource_id(self, ontology, concept_id):
        return 'http://purl.bioontology.org/ontology/%s/%s' % (
            ontology, concept_id)

    def _get_class_cui(self, ontology, concept_id):
        resource_id = self._get_resource_id(ontology, concept_id)
        resource_id = quote_plus(resource_id)
        m = self._get_json((REST_URL + '/ontologies/%s/classes/%s') % (
            ontology, resource_id))
        return set(m.get('cui', []))

    def _get_by_class_cui(self, concept_id, ontology):
        m = self._get_json(REST_URL + "/search?ontologies=%s&cui=%s" % (
            ontology, concept_id))
        results = [c['@id'].rsplit("/", 1)[1] for c in m.get('collection', [])]
        return set(results)

    def map_concept_id(self, source_terminology,
                       target_terminology, source_concept_id):
        cached = self.get_cached(source_terminology=source_terminology,
                                 target_terminology=target_terminology,
                                 source_concept_id=source_concept_id)
        if cached is not None:
            return cached
        if source_terminology == Terminology.UMLS_CUI:
            candidates = self._get_by_class_cui(source_concept_id,
                                                target_terminology.value)
        elif target_terminology == Terminology.UMLS_CUI:
            candidates = self._get_class_cui(source_terminology.value,
                                             source_concept_id)
        else:

            resource_id = self._get_resource_id(source_terminology.value,
                                                source_concept_id)
            resource_id = quote_plus(resource_id)
            m = self._get_json((REST_URL + '/ontologies/%s/classes/%s'
                                           '/mappings') % (
                                   source_terminology.value, resource_id))
            candidates = set()
            if isinstance(m, dict):
                status = m.get('status')
                print("STATUS = %s\nerror: %s" % (
                    status,
                    " ".join(m.get('errors', []))
                ))
            else:
                for result in m:
                    if not isinstance(result, dict):
                        print("Invalid response: %s" % result)
                    if result.get('source', None) != "CUI":
                        continue
                    for c in result.get('classes', []):
                        cid = c.get('@id', None)
                        if '/%s/' % target_terminology.value in cid:
                            cid = cid.rsplit('/', 1)[1]
                            candidates.add(cid)
        candidates = Terminology.validate(candidates, target_terminology)

        self.add_cached(source_terminology, target_terminology,
                        source_concept_id,
                        candidates)
        return candidates
