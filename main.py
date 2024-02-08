## Japanese Names
### Imports
import pandas
import re
import numpy
import random
import json
import requests
import glob
import os
import pickle5 as pickle
import scipy
import pymannkendall
from statsmodels.tsa.stattools import adfuller
from plotnine import *
import rstats
import pyperclip
import regex

random.seed(21)
### Settings
'''
- `SOURCE_TYPE` ... symbol source as defined in the CSV file, either `graphical_form`, `vocal_form` or `both` (concatenated).

- `VECTORIZATION_TYPE` ... how the name is vectorized, possibilities: `bag_of_symbols` convert the name into classical _bag-of-words_ of used symbols in the name,  
`symbols_semantics` converts the name by merging semantic embeddings of the single symbols of the name (see `SEMANTIC_MERGE_FUNCTION`).

- `PREPROCESS_TYPE` ... `None` or: names preprocessing before vectorization, possibilities: 
    - `ngrams` create symbol n-grams (set `n` as integer to `PREPROCESS_PARAMETER`), 
    - `letters` gets specific letters of the name (list of values `first`, `last`, `prelast`, multiple gets combined, set into `PREPROCESS_PARAMETER`).

- `SEMANTIC_MERGE_FUNCTION` ... defines numpy function to merge multiple embeddings into one, typically `numpy.mean/max/min`.
'''

ALPHA                   = 0.05
SOURCE_TYPE             = "vocal_form"          # graphical_form, vocal_form, both
VECTORIZATION_TYPE      = "len"                     # bag_of_symbols, symbols_semantics, len
PREPROCESS_TYPE         = None                 # None, ngrams, letters
PREPROCESS_PARAMETER    = None                   # "first", "last", ["last"] for SOURCE_TYPE = both, ... or blocks of ["first", "last", "prelast"], n for n-grams as 2/3/4, single_letter, None, ...
SEMANTIC_MERGE_FUNCTION = lambda embeddings: numpy.max(embeddings, axis=0)
LOGIT_OPTIM_METHOD      = "cg"                      # https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.Logit.fit.html
MINIMAL_FEATURE_COUNT_THRESHOLD = 3
ROUND_DEC = 2
EMBEDDING_SERVER_URL = " -- FILL SERVER URL HERE --"


def to_unit_vector(v):
    return v/(numpy.linalg.norm(v, 2) + 0.000001)

def nice_number(v):
    return format(v, '.2g').replace("e", "×10^")

def nround(v):
    return numpy.round(v, ROUND_DEC)

def len_of_filter(filter_function, items_list):
    return len(list(filter(filter_function, items_list)))

def format_table(df):
    s = df.to_csv(sep="\t")
    s = re.sub(r"\b0\.0\b", "0", s)
    s = re.sub(r"\b0\.", ".", s)
    return s

def get_embeddings(words):
    embeddings_dictionare = {}
    words = list( set(words) )
    r = requests.post(EMBEDDING_SERVER_URL, data=json.dumps({"language": "ja", "words" : words}))
    if r.ok:
        embeddings_dictionare = json.loads(r.content)["embeddings"]
        for word in embeddings_dictionare.keys():
            embedding = embeddings_dictionare[word]
            embeddings_dictionare[word] = to_unit_vector( numpy.array(embedding, dtype="float32") )

    return embeddings_dictionare

def prepare_data(df, column_name, postprocess_function=None):
    result = df[column_name].dropna().unique()
    if postprocess_function:
        result = [ postprocess_function(s) for s in result ]
    return list(result)

vocal_fix_substitutions = {}
AVAILABLE_SUBSTITUTIONS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def fix_vocalic_name(name):
    global vocal_fix_substitutions
    
    doubles = re.findall("((.)([ぁぃぅぇぉゃゅょァィゥェォャュョ]))", name)
    for doublet_match in doubles:
        doublet = doublet_match[0]
        if not doublet in vocal_fix_substitutions:
            assert len(AVAILABLE_SUBSTITUTIONS) > len(vocal_fix_substitutions)
            vocal_fix_substitutions[doublet] = AVAILABLE_SUBSTITUTIONS[len(vocal_fix_substitutions)]
        name = name.replace(doublet, vocal_fix_substitutions[doublet])
        
    return name

def make_compounds(source_graph, source_vocal):
    """Joins graphical and vocal form into one string."""
    compounds = zip( source_graph, source_vocal )
    compounds = list( set(compounds) )
    return compounds

def filter_lengths_for_tuples(tuples_source, min_length = 1):
    output = [ (a, b) if min(len(a), len(b)) > min_length else None for a, b in tuples_source ]
    output = [ o for o in output if o ]
    return output

### Preprocessing
# Preprocess the names: ngramize, get given letters, ... or None

def ngramize(word, n=2, add_border_info=True):
    """Creates symbol n-grams from a given `word`."""
    output = []
    if len(word) < n:
        output.append(word)
    else:
        for i in range(0, len(word)-n+1):
            output.append( word[i:i+n] )

    if add_border_info:
        output[0] = "^" + output[0]
        output[-1] = output[-1] + "$"
    return output

def get_letters(word, which):
    output = ""
    return_list = type(which) == list

    if isinstance(word, tuple):
        output = get_letters(word[0], which) + get_letters(word[1], which)
        output = "".join(output)
    else:
        if which == "single_letter":
            if len(word) != 1:
                raise ValueError("Words with length > 1 are contained, something went wrong!")

            output = word

        if "first" in which:
            output += word[0]
        
        if "prelast" in which:
            if len(word) > 1:
                output += word[-2]

        if "last" in which:
            output += word[-1]

    return [output] if return_list else output

def get_single_symbols(name, dummy=None):
    return list(name)

def vectorize_symbols(name, symbol_to_vector_position, use_frequencies=True):
    vector = numpy.zeros(len(symbol_to_vector_position))
    for symbol in name:
        if symbol in symbol_to_vector_position:
            position = symbol_to_vector_position[symbol]
            if use_frequencies:
                vector[position] += 1
            else:
                vector[position] = 1
    return vector

def vectorize_semantics_symbols(name, symbol_to_embedding):
    embeddings = [ symbol_to_embedding[s] if s in symbol_to_embedding else numpy.zeros(300) for s in name ]
    embeddings = numpy.array(embeddings)
    vector = SEMANTIC_MERGE_FUNCTION(embeddings)
    return vector


def to_alphabet_mask(word):
    """Converts string くるみ   ソフィア   なの香   花子 => III KKKK IIH HH."""
    word = word.strip()
    mask = regex.sub(r'\p{IsHan}', "H", word, re.UNICODE)
    mask = regex.sub(r'\p{IsHira}', "I", mask, re.UNICODE)
    mask = regex.sub(r'\p{IsKatakana}', "K", mask, re.UNICODE)
    mask = regex.sub(r'ー', "K", mask, re.UNICODE)
    return mask

def is_mask_only_han(mask):
	return all([ s == "H" for s in mask ])

def is_mask_only_katakana(mask):
	return all([ s == "K" for s in mask ])

def is_mask_only_hiragana(mask):
	return all([ s == "I" for s in mask ])

def is_mask_han_hira_mix(mask):
	return "H" in mask and "I" in mask

def is_mask_han_katakana_mix(mask):
	return "H" in mask and "K" in mask



###################################################################################################

def evaluate_dataset(df_m, df_f, verbose=False):
    output = {}

    male_graph = prepare_data(df_m, "M_GRAPH")
    male_vocal = prepare_data(df_m, "M_VOC", fix_vocalic_name)

    female_graph = prepare_data(df_f, "F_GRAPH")
    female_vocal = prepare_data(df_f, "F_VOC", fix_vocalic_name)

    male_both   = make_compounds(df_m["M_GRAPH"], map(fix_vocalic_name, df_m["M_VOC"]))
    female_both = make_compounds(df_f["F_GRAPH"], map(fix_vocalic_name, df_f["F_VOC"]))

    male_values   = [ len(male_graph),   len(male_vocal),   len(male_both) ]
    female_values = [ len(female_graph), len(female_vocal), len(female_both)]
    
    ### SUMMARY
    output["summary_info"] = pandas.DataFrame(
                                [male_values, female_values], 
                                columns=["Graphical", "Vocal", "Both (Combination)"], 
                                index=["Male", "Female"]
                            )

    if verbose:
        print(output["summary_info"])
        output["summary_info"].to_clipboard()

        print("Empirical vocal substitution:")
        print(vocal_fix_substitutions)



    ### Ortography summary  #######################################################################
    male_graph_alpha_mask   = [ to_alphabet_mask(name) for name in male_graph ]
    female_graph_alpha_mask = [ to_alphabet_mask(name) for name in female_graph ]
    ortography_results = {}
    
    def get_ortography_stats(ortography_selector):        
        females_len = len_of_filter(ortography_selector, female_graph_alpha_mask)
        males_len = len_of_filter(ortography_selector, male_graph_alpha_mask)


        table = [
            [females_len, len(female_graph)],
            [males_len,   len(male_graph)]
        ]

        fisher_pvalue, fisher_odds, fisher_odds_ci  = rstats.calculate_fisher_exact_test_R( numpy.array(table).flatten().tolist() )

        ortography_results[ortography_selector.__name__] =  (
            table[1][0], 
            table[0][0], 
            nround(fisher_odds),
            nround(fisher_odds_ci),
            nice_number(fisher_pvalue),
            "YES" if fisher_pvalue < ALPHA/5 else "NO"
        )

        return males_len, females_len

    omnibus = [
        get_ortography_stats(is_mask_only_han),
        get_ortography_stats(is_mask_only_hiragana),
        get_ortography_stats(is_mask_only_katakana),
        get_ortography_stats(is_mask_han_hira_mix),
        get_ortography_stats(is_mask_han_katakana_mix)
    ]

    chisq, df, p, mcfadden = rstats.calculate_simulated_chisq(list( zip(*omnibus) ))
    phi = numpy.sqrt(chisq/(len(male_graph)+len(female_graph)))

    ortography_results_df = pandas.DataFrame(ortography_results, index=["males", "femals", "odds", "odds-ci", "pvalue", "signf"]).transpose()
    if verbose:
        print("------------------------------------------------------------------------------")
        print("Ortography results:")
        print(f"ChisSq: {chisq}, p-value: {nice_number(p)}, alpha': {0.05/5}, phi: {phi}")
        print("Detailed:")
        print(ortography_results_df)
        ortography_results_df.to_clipboard()
        print("------------------------------------------------------------------------------")
        #input("... press [enter] to continue stats...")

    ## Vocal Len odds analysis  ###################################################################
    if SOURCE_TYPE == "vocal_form":
        m = male_vocal
        f = female_vocal
    elif SOURCE_TYPE == "graph_form":
        m = male_graph
        f = female_graph
    else:
        raise ValueError("Please choose names source!")

    male_lens   = list( map(len, m) )      ###<<< male_graph vs. male_phono
    female_lens = list( map(len, f) )    ###<<< graph vs phono
    min_len, max_len = min( male_lens + female_lens ), max( male_lens + female_lens )

    counts = {
        i : [ 
            [female_lens.count(i), len(female_lens)], 
            [male_lens.count(i), len(male_lens) ]
        ] 
        for i in range(min_len, max_len+1)
    }

    ortography_phonology_lens_df = pandas.DataFrame(columns=["Len", "M", "F", "Odds", "OddsCI","p-value", "Significant"])
    for count_table_len, table in counts.items():
        fisher_p_value, fisher_odds, fisher_ci = rstats.calculate_fisher_exact_test_R( numpy.array(table).flatten() )
        signif_info = str(fisher_p_value < ALPHA/len(counts))
        ortography_phonology_lens_df = ortography_phonology_lens_df.append(
            pandas.Series([
                count_table_len, table[1][0], table[0][0], nround(fisher_odds), nround(fisher_ci), nice_number(fisher_p_value), signif_info
            ], index=ortography_phonology_lens_df.columns), 
            ignore_index=True)

    if verbose:
        print("------------------------------------------------------------------------------")
        print(ortography_phonology_lens_df)
        ortography_phonology_lens_df.to_clipboard()
        print("..............................................................................")
        #input("... press [enter] to continue stats...")

    ## SOURCE PICK  ###############################################################################
    source_male   = None
    source_female = None

    if SOURCE_TYPE == "graphical_form":
        source_male   = male_graph
        source_female = female_graph
    if SOURCE_TYPE == "vocal_form":
        source_male   = male_vocal
        source_female = female_vocal
    if SOURCE_TYPE == "both":
        source_male   = male_both
        source_female = female_both

    if source_male is None or source_female is None:
        raise ValueError("Please choose names source!")

    filter_one_symbos = any([
        SOURCE_TYPE == "graphical_form" and not VECTORIZATION_TYPE == "len",
        PREPROCESS_TYPE == "letters" and "prelast" in PREPROCESS_PARAMETER,
        PREPROCESS_TYPE == "letters" and "last" in PREPROCESS_PARAMETER
    ])

    if PREPROCESS_TYPE == "letters" and PREPROCESS_PARAMETER == "single_letter":
        source_male   = list( filter(lambda name: len(name) == 1, source_male) )
        source_female = list( filter(lambda name: len(name) == 1, source_female) )
        filter_one_symbos = False
    else:
        if filter_one_symbos:
            if SOURCE_TYPE == "both":
                source_male   = filter_lengths_for_tuples(source_male)
                source_female = filter_lengths_for_tuples(source_female)
            else:
                source_male   = list( filter(lambda name: len(name) > 1, source_male) )
                source_female = list( filter(lambda name: len(name) > 1, source_female) )

    if verbose:
        if filter_one_symbos:
            print(" *** NAME LENGTHS WERE FILTERED (>1) ***")

        lengths = list( map(len, source_male + source_female) )
        print(f"Available Names Lengths: { numpy.min(lengths) } -- { numpy.max(lengths) }")

    preprocess = None
    if PREPROCESS_TYPE == None:
        preprocess = get_single_symbols
    if PREPROCESS_TYPE == "ngrams":
        preprocess = ngramize
    if PREPROCESS_TYPE == "letters":
        preprocess = get_letters

    if preprocess:
        source_male   = [ preprocess(name, PREPROCESS_PARAMETER) for name in source_male ]
        source_female = [ preprocess(name, PREPROCESS_PARAMETER) for name in source_female ]

    train_names = source_male + source_female
        
    if verbose:
        print(f"Preprocessing example ({PREPROCESS_TYPE}, {PREPROCESS_PARAMETER})")
        print(source_male[0:5])
        print("... building bag-of-symbols")

    ## VECTORIZE NAMES
    symbol_to_vector_position = {}
    symbol_by_position = []

    for name in train_names:
        for symbol in name:
            if not symbol in symbol_to_vector_position:
                symbol_to_vector_position[symbol] = len(symbol_to_vector_position)
                symbol_by_position.append(symbol)

    if verbose:
        print(f"... getting symbol vectorizations for n = {len(symbol_to_vector_position)} symbols")

    if VECTORIZATION_TYPE == "bag_of_symbols":
        f_vectorize = lambda word: vectorize_symbols(word, symbol_to_vector_position)
    elif VECTORIZATION_TYPE == "symbols_semantics":
        f_vectorize = lambda word: vectorize_semantics_symbols(
            word, 
            get_embeddings(list(symbol_to_vector_position.keys()))
        )
    elif VECTORIZATION_TYPE == "len":
        f_vectorize = len
    else:
        raise ValueError(f"Invalid vectorization option {VECTORIZATION_TYPE}")

    if verbose:
        print("... vectorizing:")
        print(f"Total: {len(source_male)  + len(source_female)} names, males: {len(source_male)}, females: {len(source_female)}")

    all_X = numpy.array( list(map(f_vectorize, train_names)), dtype="float32" )
    all_y = numpy.array( [0] * len(source_male) + [1] * len(source_female) )

    feature_selector = all_X.sum(axis=0) >= MINIMAL_FEATURE_COUNT_THRESHOLD
    all_X = all_X[:, feature_selector ]

    if all_X.shape[1] == 0:
        if verbose:
            print("Features seem unique! See vectorization and features.")
        raise ValueError("No features with the minimal frequency found! Features seem unique!")

    key_names = None
    if VECTORIZATION_TYPE == "bag_of_symbols":
        key_names = numpy.array(symbol_by_position)
        key_names = key_names[feature_selector]
    else:
        key_names = VECTORIZATION_TYPE

    output["all_data_shape_X"] = all_X.shape
    output["all_data_shape_y"] = all_y.shape

    if verbose:
        print(f"All shape: {all_X.shape}")
        print(f"... train X shape: {all_X.shape}, y shape: {all_y.shape}")
        print("... data X example:")
        print(all_X[0:2])
        print("... data y example:")
        print(all_y[0:2])
        print("Bag of symbols example:")
        print(symbol_by_position[0:5])
        print("")
        print("==============================================================================")
        print(f"{SOURCE_TYPE} {PREPROCESS_TYPE} {PREPROCESS_PARAMETER}")
        print("==============================================================================")

    if VECTORIZATION_TYPE == "len":
        table = pandas.crosstab( all_X.flatten(), all_y).transpose()
        males = table.loc[0].to_numpy()
        females = table.loc[1].to_numpy()
        chi2, df, pvalue, mcfadden = rstats.calculate_simulated_chisq( [males, females] )
        phi = numpy.sqrt(chi2 / sum(males + females) )

        if verbose:
            print(f"... Males vs. Females differnece:    Simulated p-value chi2 = {nround(chi2)}, p = {pvalue}, phi = {nround(phi)}, McF = {nround(mcfadden)}")
            print(table)
            table.to_clipboard()

        import statsmodels.api as sm    
        import compare_auc_delong_xu
        from sklearn.linear_model import LogisticRegression

        all_X = sm.add_constant(all_X)
        logit_mod = sm.Logit(all_y, all_X)
        model = logit_mod.fit(maxiter=9999, method=LOGIT_OPTIM_METHOD, disp=verbose)
        all_yhat = model.predict(all_X)
        auc, auc_var = compare_auc_delong_xu.delong_roc_variance(all_y, all_yhat)
        auc_ci_l = auc - numpy.sqrt(auc_var) * 1.96
        auc_ci_u = auc + numpy.sqrt(auc_var) * 1.96
        auc_ci = [auc_ci_l, auc_ci_u]

        output["score"] = phi #auc
        output["score_var"] = -1 #auc_var
        output["n"]  = len(all_X)
        output["mcfadden"] = mcfadden
        
        if verbose:
            print( model.summary() )
            print( f"    Coef: {nround(model.params[1])}    Odds: {nround(numpy.exp(model.params[1]))}    CI: {nround( numpy.exp( model.conf_int()[1] ))}    pvalue: {model.pvalues[1]}" )
            print( f"    AUC: {nround(auc)}, CI: {nround(auc_ci)}")
        
        return output

    ### Feature map
    males_count = (all_y == 0).sum()
    females_count = (all_y == 1).sum()
    male_counts = all_X[ all_y == 0 ].sum(axis=0, dtype="int")
    female_counts = all_X[ all_y == 1 ].sum(axis=0, dtype="int")

    results = pandas.DataFrame(columns=["Symbol", "Males", "Females", "Total", "Odds", "CI", "p-value"])
    for key_name, male_count_feature, female_count_feature in zip(key_names, male_counts, female_counts):
        #               Male     Female
        # Has Feature    32      21
        # Do not have    92      200
        table = numpy.array([ [female_count_feature, male_count_feature], 
                                [females_count, males_count] ])
        sc_odds, sc_pvalue = scipy.stats.fisher_exact(table, alternative='two-sided')
        pvalue, odds, odds_ci = rstats.calculate_fisher_exact_test_R( table.flatten().tolist() )

        row = {"Symbol" : key_name, 
                "Males" : male_count_feature, 
                "Females" : female_count_feature, 
                "Total" : male_count_feature+female_count_feature, 
                "Odds" : nround(odds),
                "CI" : str(list( nround(odds_ci) )),
                "CI_L" : odds_ci[0],
                "CI_U" : odds_ci[1],
                "p-value" : pvalue}
        results = results.append(row, ignore_index=True)

    n = len(key_names)
    corrected_alpha = ALPHA / n
    corrected_alpha =   0.01
    results = results[ results["p-value"] <= corrected_alpha ] ### EXT TABULKA
    results = results.sort_values(["Odds", "Males", "Females"], ascending=[True, False, True])
    results = results.astype({"Males" : "int", "Females" : "int", "Total" : "int"})
    output["symbols_results"] = results

    ## All features
    table = numpy.array([male_counts, female_counts])
    chi2, df, pvalue, mcfadden = rstats.calculate_simulated_chisq(table, simulate_p_value=True) # [Hope 1968 method] Hope, A. C. A. (1968) A simplified Monte Carlo significance test procedure. J. Roy, Statist. Soc. B 30, 582-598.

    n = numpy.sum(table)
    phi = numpy.sqrt(chi2/n)
    number_of_features = len(results)
    
    output["p"] = pvalue
    output["score"] = phi
    output["n"] = n
    output["score_var"] = -1 # Chi2Dist(number_of_features-1)/numpy.sqrt(males_count+females_count)
    output["mcfadden"] = mcfadden

    if verbose:
        print(f"... Names total x number of symbols examined = { all_X.shape }")
        print(f"... Males vs. Females differnece:    Simulated p-value chi2 = {nround(chi2)}, p = {pvalue}, phi = {nround(phi)}, mcf = {nround(mcfadden)}")
        print(f"... features n = {len(key_names)}, correcting ALPHA {ALPHA} = {corrected_alpha} = {nice_number(corrected_alpha)}")
        print(f"... significant m = { len(results) }")
        print(f"... filtered names lengths (>1): {filter_one_symbos}\n")

        results_printable = results.drop(columns=["CI_L", "CI_U"])
        results_printable["p-value"] = results_printable["p-value"].apply( lambda v: nice_number(v) )
        s = format_table(results_printable)
        pyperclip.copy(s)
        #results_printable.to_clipboard()
        print(s)

    ## Plot
    d = pandas.DataFrame( index=key_names )
    d["Symbol"] = key_names
    d["Male"]   = male_counts / males_count * 100
    d["Female"] = female_counts / females_count * 100
    d["Significant"] = "No"
    d["Odds"] = 1
    d["LogOdds"] = 1
    d.loc[ results["Symbol"], "Significant" ] = "Yes"

    for index, significant_row in results.iterrows():
        odds = significant_row["Odds"]
        if odds == 0:
            odds = 100
        if odds < 1:
            odds = 1/odds

        d.loc[significant_row["Symbol"], "Odds"] = odds
        d.loc[significant_row["Symbol"], "LogOdds"] = numpy.log(odds) + 1

    output["feature_map"] = d
    if verbose:
        [ggplot(d, aes(x="Male", y="Female")) +                                                                                                  
            geom_abline(intercept=0, slope=1, linetype="dashed") + 
            geom_text(aes(label="Symbol", colour="Significant"), family="MS Gothic") + 
            theme_bw() + 
            xlab("Males [%]") + 
            ylab("Females [%]") + 
            coord_fixed(ratio=1) + 
            theme(aspect_ratio=1) + 
            scale_fill_manual(values = {"Yes" : "black", "No" : "yellow"} ) ]


    ## Plot 2
    results["CI_U_CLIPPED"] = results["CI_U"].clip(0, 20)
    ggplot(results, aes("CI_L", "CI_U_CLIPPED")) + geom_errorbar(aes(ymin="CI_L", ymax="CI_U_CLIPPED"), width=0.2)

    return output

###################################################################################################
## MAIN FUNCTIONS   ###############################################################################
###################################################################################################

def evaluate_all():
    df_m = pandas.read_csv("data/names_m_new.txt", encoding="UTF-8", sep="\t")
    df_f = pandas.read_csv("data/names_f_new.txt", encoding="UTF-8", sep="\t")

    evaluate_dataset(df_m, df_f, verbose=True)


def evaluate_by_years(save_results=True):
    males_data   = glob.glob("data/per_year/*_m.txt")
    females_data = glob.glob("data/per_year/*_f.txt")
    
    ## test years match
    unified_males_data_test = [ unified_name.replace("_m", "_f") for unified_name in males_data ]
    difference = set(unified_males_data_test).difference( set(females_data) )
    if len(difference):
        raise ValueError(f"... for per-year data, there are few differences in: {str(difference)}")

    year_names = [ re.search(r"\d{4}", os.path.basename(file_name)).group() for file_name in males_data ]
    results = { key : [] for key in year_names }
    
    for male_file_name, female_file_name, year in zip(males_data, females_data, year_names):
        print("\n" + male_file_name)
        df_m = pandas.read_csv(male_file_name,   encoding="UTF-8", sep="\t", header=None, names=["M_GRAPH", "M_VOC"])
        df_f = pandas.read_csv(female_file_name, encoding="UTF-8", sep="\t", header=None, names=["F_GRAPH", "F_VOC"])

        results[year] = evaluate_dataset(df_m, df_f, verbose=True)

    report_by_years_evaluation(results)


def report_by_years_evaluation(results):
    print("")
    print("==============================================================================")
    print(f" BY YEARS: {SOURCE_TYPE} {PREPROCESS_TYPE} {PREPROCESS_PARAMETER}")
    print("==============================================================================")
    print("")
    
    '''years = result.keys()
    for year in years:
        counts = result[year]["summary_info"][0]
        print(year)
        print(counts)
        print("")'''

    info = pandas.DataFrame( { key : [results[key]["score"], results[key]["score_var"], results[key]["n"]] for key in results.keys() }, 
                                index=["score", "var", "n"] )

    s = format_table( info.transpose().drop(columns=["var"]).transpose().round(ROUND_DEC) )
    print(s)
    pyperclip.copy(s)

    raw_value = info.loc["score"].values
    raw_value_var = info.loc["var"].values
    
    ## Method 1:
    spearman = scipy.stats.spearmanr(range(0, len(raw_value)), raw_value)
    print("")
    print(f"Spearman Correlation {nround(spearman.correlation)} p-value: {nround(spearman.pvalue)}")

    ## Method 2:
    ## Mann-Kendall Trend Test
    trend = pymannkendall.original_test(raw_value)
    print("")
    print(f"Mann-Kendall Trend Test: {trend.trend} s: {nround(trend.s)}  p-value: {nround(trend.p)}")


def main(task_name):
    if task_name == "evaluate_all":
        evaluate_all()
    elif task_name == "evaluate_by_years":
        evaluate_by_years()


if __name__ == "__main__":
    main("evaluate_all")
    while input("... write [p] to proceed.") != "p":
        pass

    main("evaluate_by_years")