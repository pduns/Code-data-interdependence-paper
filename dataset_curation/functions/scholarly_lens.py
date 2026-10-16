#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import json
import sys
import requests
from datetime import datetime
from requests.exceptions import ChunkedEncodingError, Timeout, ConnectionError
from urllib3.exceptions import ProtocolError
import pandas as pd 
import re

def expand_term(term: str, replacement_list):
    """
    Given a term and a list of replacement groups (each as 'a; b; c'),
    return all variants created by substituting any word/phrase in a group
    by any other word/phrase in that same group. All replacements
    are applied combinatorially.
    """
    variants = {term}

    for group in replacement_list:
        # split and strip
        alternatives = [p.strip() for p in group.split(';') if p.strip()]
        if not alternatives:
            continue

        # build a regex that matches any of the alternatives
        pattern = re.compile('|'.join(re.escape(a) for a in alternatives))

        new_variants = set(variants)
        for v in variants:
            if not pattern.search(v):
                continue

            # for *each* alternative, replace all occurrences in v
            for alt in alternatives:
                new_variants.add(pattern.sub(alt, v))

        variants = new_variants

    return sorted(variants)


def build_search_query_scholarly_lens(
    search_terms1,
    search_terms2=None,
    search_terms3=None,
    *,
    year_min=2010,
    size=1000,
    open_access=True,           # True/False to filter; None = don't filter by OA
    search_title=True,
    search_abstract=True,
    search_fulltext=True,
    bundle1_operator="OR",      # "OR" | "AND" within bundle 1
    bundle2_operator="OR",      # "OR" | "AND" within bundle 2 (if provided)
    bundle3_operator="OR",      # "OR" | "AND" within bundle 3 (if provided)
    publication_types=None,
    include=None,               # fields to return (always included in output)
):
    """
    Build an Elasticsearch query from up to three bundles of terms.

    Accepts either a single string or a list of strings for each bundle.
    Terms are used EXACTLY as provided.
    """

    # --- Normalize inputs: allow single string or list ---
    def normalize(x):
        if x is None:
            return None
        if isinstance(x, str):
            return [x]
        return list(x)

    search_terms1 = normalize(search_terms1)
    search_terms2 = normalize(search_terms2)
    search_terms3 = normalize(search_terms3)

    if publication_types is None:
        publication_types = [
            "journal article",
            "conference proceedings",
            "conference proceedings article",
            "report",
        ]

    if include is None:
        include = [
            "lens_id",
            "title",
            "publication_type",
            "abstract",
            "keywords",
            "source",
            "fields_of_study",
            "year_published",
            "authors",
            "funding",
            "open_access",
            "external_ids",
            "references",
        ]

    # --- Determine fields ---
    fields = []
    if search_title:
        fields.append("title")
    if search_abstract:
        fields.append("abstract")
    if search_fulltext:
        fields.append("fulltext")
    if not fields:
        raise ValueError("Enable at least one of search_title, search_abstract, or search_fulltext.")

    # --- Clause builder ---
    def make_clauses(terms):
        clauses = []
        for term in terms or []:
            t = term.strip()
            if not t:
                continue
            for f in fields:
                clauses.append({"query_string": {"query": f"{f}:{t}"}})
        return clauses

    def wrap_bundle(clauses, operator):
        op = (operator or "OR").upper()
        if op not in {"OR", "AND"}:
            op = "OR"
        if not clauses:
            return {"bool": {"must_not": [{"match_all": {}}]}}
        return {"bool": {"should": clauses}} if op == "OR" else {"bool": {"must": clauses}}

    # --- Bundle 1 ---
    b1_clauses = make_clauses(search_terms1)
    if not b1_clauses:
        raise ValueError("search_terms1 must contain at least one non-empty term.")
    b1 = wrap_bundle(b1_clauses, bundle1_operator)

    # --- Bundles 2 & 3 ---
    must_blocks = [b1]

    if search_terms2:
        b2_clauses = make_clauses(search_terms2)
        if b2_clauses:
            must_blocks.append(wrap_bundle(b2_clauses, bundle2_operator))

    if search_terms3:
        b3_clauses = make_clauses(search_terms3)
        if b3_clauses:
            must_blocks.append(wrap_bundle(b3_clauses, bundle3_operator))

    # --- Filters ---
    filters = [
        {"range": {"year_published": {"gte": year_min}}},
        {"terms": {"publication_type": publication_types}},
    ]
    if open_access is not None:
        filters.append({"term": {"is_open_access": open_access}})

    # --- Final query body ---
    body = {
        "query": {"bool": {"must": must_blocks + filters}},
        "size": size,
        "include": include,
    }

    return body


def lens_scroll_search(
    api_key: str,
    request_body: dict,
    *,
    use_max_scrolls: bool = True,
    max_scrolls: int = 10,
    path: str = "path",
    filename: str = "name",
    date: str = "date"
):
    """
    Simple Lens API scroll function with optional scroll limit.

    IMPORTANT:
    - 'include' MUST be specified by the caller in `request_body` (next to 'size').
      This function does NOT modify or inject 'include'.

    Parameters
    ----------
    api_key : str
        Your Lens API key.
    request_body : dict
        The JSON body for your initial POST request.
        (e.g. {'query': {...}, 'size': 1000, 'include': [...]} )
    use_max_scrolls : bool
        Whether to stop after `max_scrolls` iterations.
    max_scrolls : int
        Maximum number of scroll requests if use_max_scrolls=True.

    Returns
    -------
    list[dict]
        All data records returned by the API.
    """
    url = "https://api.lens.org/scholarly/search"
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    data_list = []
    scroll_count = 0
    scroll_id = None

    # Make a shallow copy so we don't mutate the caller's body
    initial_payload = dict(request_body)
    # Ensure the initial request has a scroll window
    initial_payload["scroll"] = initial_payload.get("scroll", "5m")

    while True:
        print(f"Scroll #{scroll_count + 1}")
        try:
            if scroll_id:
                # For scroll calls, only send scroll_id + scroll window.
                payload = {"scroll_id": scroll_id, "scroll": "5m"}
            else:
                payload = initial_payload

            # Use json=... so requests sets headers/encoding correctly.
            response = requests.post(url, json=payload, headers=headers)

        except (ChunkedEncodingError, ProtocolError, Timeout, ConnectionError) as e:
            print(f"Connection error: {e}. Retrying in 60s ...")
            time.sleep(60)
            continue
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying in 60s ...")
            time.sleep(60)
            continue

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "10"))
            print(f"429 Too Many Requests. Waiting {retry_after}s ...")
            time.sleep(retry_after)
            continue
        elif response.status_code == 204:
            print("No more results.")
            break
        elif response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            break

        try:
            json_response = response.json()
        except Exception as e:
            print(f"Error reading JSON: {e}")
            break

        results = json_response.get("data", [])
        if not results:
            print("No more data.")
            break

        data_list.extend(results)
        scroll_id = json_response.get("scroll_id")
        scroll_count += 1
        print(f"Fetched {len(results)} records. Total so far: {len(data_list)}")

        if use_max_scrolls and scroll_count >= max_scrolls:
            print("Reached max scroll limit.")
            break

        if not scroll_id:
            # defensive: if API doesn't return a new scroll_id, we're done
            print("No scroll_id returned. Stopping.")
            break

    print(f"✅ Finished. Total records: {len(data_list)}")

    # Save raw data
    out_path = path + r'raw_data\\' + date + r'_scholarly_' + filename + '.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False)

    print('data_list saved as: ' + out_path)
    return data_list





def clean_scholarly_lens(data_list, path = 'path', filename = 'name', date = 'date'):
    lens_ids = []
    titles = []
    publication_types = []
    abstracts = []
    keywords = []
    asjc_subjects = []
    fields_of_study = []
    dates = []
    affiliations = []
    affiliation_ids = []
    affiliation_countries = []
    affiliation_fundrefs = []
    funding_sources = []
    references = []
    open_access_urls = []
    open_access_pdf_urls = []
    dois = []
    magids = []
    
    	
    
    
    for publication in data_list: 
        affiliations_str = ''
        affiliation_ids_str = ''
        affiliation_countries_str = ''
        affiliation_fundings_str = ''
        
        nr = 0
    
        lens_ids.append(publication['lens_id'])
    
        titles_placeholder = ''
        if 'title' in publication.keys():
            titles_placeholder = publication['title']
        titles.append(titles_placeholder)
        
        publication_types_placeholder = ''
        if 'publication_type' in publication.keys():
            publication_types_placeholder = publication['publication_type']
        publication_types.append(publication_types_placeholder)
    
        dates_placeholder = ''
        if 'year_published' in publication.keys():
            dates_placeholder = publication['year_published']
        dates.append(dates_placeholder)
        
        
        abstract_placeholder = ''
        if 'abstract' in publication.keys():
            abstract_placeholder = publication['abstract']
        abstracts.append(abstract_placeholder)
        
        keywords_placeholder = ''
        if 'keywords' in publication.keys():
            keywords_placeholder = publication['keywords']
        keywords.append(keywords_placeholder)
    
        asjc_subjects_placeholder = ''
        if 'source' in publication.keys(): 
            if 'asjc_subjects' in publication['source'].keys():
                asjc_subjects_placeholder = publication['source']['asjc_subjects']
        asjc_subjects.append(asjc_subjects_placeholder)
    
        fields_of_study_placeholder = ''
        if 'fields_of_study' in publication.keys():
            fields_of_study_placeholder = publication['fields_of_study']
        fields_of_study.append(fields_of_study_placeholder)
    

        doi_ph = ''
        magid_ph = ''
        if 'external_ids' in publication.keys():
            for id_type in publication['external_ids']:
                if id_type['type'] == 'doi':
                    doi_ph = id_type['value']
                if id_type['type'] == 'magid':
                    magid_ph = id_type['value']
        dois.append(doi_ph)
        magids.append(magid_ph)

        funding_placeholder = ''
        if 'funding' in publication.keys():
            funding_placeholder = publication['funding']
        funding_sources.append(funding_placeholder)

        within_pub_references = ''
        if 'references' in publication.keys(): 
            within_pub_references = []
            for reference in publication['references']: 
                within_pub_references.append(reference['lens_id'])
            
        references.append(within_pub_references)

        open_access_urls_placeholder = ''
        open_access_pdf_urls_placeholder = ''
        if 'open_access' in publication.keys(): 
            if 'locations' in publication['open_access'].keys(): 
                if 'landing_page_urls' in publication['open_access']['locations'].keys():
                    open_access_urls_placeholder = publication['open_access']['locations']['landing_page_urls']
                if 'pdf_urls' in publication['open_access']['locations'].keys():
                    open_access_pdf_urls_placeholder = publication['open_access']['locations']['pdf_urls']
        open_access_urls.append(open_access_urls_placeholder)
        open_access_pdf_urls.append(open_access_pdf_urls_placeholder)
        
        if 'authors' in publication.keys(): 
            for author in publication['authors']: 
                ##afilliations
                affiliations_placeholder = ''
                affiliation_ids_placeholder = ''
                affiliation_countries_placeholder = ''
                affiliation_funding_placeholder = ''
                if 'affiliations' in author.keys() and author['affiliations'] != []:
                    if 'name' in author['affiliations'][0].keys():
                        affiliations_placeholder = author['affiliations'][0]['name']
                    
                    #affiliation ids (affiliation id and funding reference) 
                    if 'ids' in author['affiliations'][0]:
                        for id_dict in author['affiliations'][0]['ids']:
                            if 'magid' in id_dict.values():
                                affiliation_ids_placeholder = id_dict['value']
                            if 'fundref' in id_dict.values():
                                affiliation_funding_placeholder = id_dict['value']
                            #break #break after one identifier was encountered (not necessary if working with lists anyway, only if I need one per author)
                        
    
                    #countries
                    if 'country_code' in author['affiliations'][0].keys():
                        affiliation_countries_placeholder = author['affiliations'][0]['country_code']
                
                #add all publication affiliation info together in string form
                if affiliations_str == '': 
                    if affiliations_placeholder != '': 
                        affiliations_str = affiliations_placeholder
                    else:
                        affiliations_str = ''    
                else: 
                    if affiliations_placeholder != '': 
                        affiliations_str = affiliations_str+ '; ' + affiliations_placeholder
                    else:
                        affiliations_str = affiliations_str
    
                if affiliation_ids_str == '': 
                    if affiliation_ids_placeholder != '': 
                        affiliation_ids_str = affiliation_ids_placeholder
                    else: 
                        affiliation_ids_str = ''
                else:
                    if affiliation_ids_placeholder != '': 
                        affiliation_ids_str = affiliation_ids_str+ '; ' + affiliation_ids_placeholder
                    else: 
                        affiliation_ids_str = affiliation_ids_str
    
                if affiliation_countries_str == '':
                    if affiliation_countries_placeholder != '': 
                        affiliation_countries_str = affiliation_countries_placeholder
                    else: 
                        affiliation_countries_str = ''
                else: 
                    if affiliation_countries_placeholder != '': 
                        affiliation_countries_str = affiliation_countries_str+ '; ' + affiliation_countries_placeholder
                    else:
                        affiliation_countries_str = affiliation_countries_str
                    
                if affiliation_fundings_str == '': 
                    if affiliation_funding_placeholder != '': 
                        affiliation_fundings_str = affiliation_funding_placeholder
                    else: 
                        affiliation_fundings_str = ''     
                else:
                    if affiliation_funding_placeholder != '': 
                        affiliation_fundings_str = affiliation_fundings_str + '; ' + affiliation_funding_placeholder
                    else:
                        affiliation_fundings_str = affiliation_fundings_str
                        
        #merge the affiliation info lists into the main df:
        affiliations.append(affiliations_str)
        affiliation_ids.append(affiliation_ids_str)
        affiliation_countries.append(affiliation_countries_str)
        affiliation_fundrefs.append(affiliation_fundings_str)
                    
    
    # export it as dictionary
    df_dict = {'lens_id': lens_ids,  'title': titles, 'publication_type': publication_types,'abstract': abstracts, 'keywords':keywords, 'asjc_subjects': asjc_subjects,
    'fields_of_study': fields_of_study,
    'dates': dates,
    'affiliations': affiliations,
    'affiliation_ids': affiliation_ids,
    'affiliations_countries': affiliation_countries,
    'fundrefs': affiliation_fundrefs,
    'funding_sources': funding_sources,
    'references': references,
    'doi': dois,
    'magid': magids,
    'open_access_urls': open_access_urls,
    'open_access_pdf_urls': open_access_pdf_urls
    }
    df_scholarly = pd.DataFrame(df_dict)
    
    #save 
    df_scholarly.to_csv(path + r'processed_data\\'+date+r'_scholarly_'+filename+'.csv')

    print('cleaned data df saved as: ' + path + r'processed_data\\'+date+r'_scholarly_'+filename+'.csv')
    
    return df_scholarly

