"""Phrasings for the gene-list recogniser, sized, and where each came from.

- REVIEW_*: written by an adversarial review of the first version, which
  fired on 14 of the 44 questions and missed or trimmed 20 of the 25
  requests. The recogniser was then rewritten against these.
- HELD_OUT_*: written after that rewrite and never tuned against. They are
  the honest measure: 15/15 and 0/20 when added.

- SECOND_REVIEW_*: from a second review, of the version that proposed
  rather than ran. It fired on 12 of 40 questions and read the wrong list
  or none for 3 of 25; the parser was rewritten again.
- HELD_OUT_3_*: written after that, never tuned against: 11/12 requests
  exact, 1/15 questions fired. The miss and a pasted table (which
  should be attached as a file) are KNOWN_LIMITS, pinned as they are.
  The misfire was later fixed, by the rule for round three's misfires.
- A third review's fresh set, on the version after that: 18/20 requests
  exact, 4/20 questions offered an analysis. The offer costs one click;
  the rate is the honest one to quote.
- THIRD_REVIEW_*: that review's fresh set, then tuned against (4
  question misfires, all two genes in a question, fixed by one rule).
- HELD_OUT_4_*: written after round three's fixes, never tuned against:
  9/10 requests exact, 0/12 questions offered. The miss is a KNOWN_LIMIT.

A positive is exact: the identifiers read, in order. A request that reads
the wrong list is a failure even if it fires.
"""

REVIEW_REQUESTS: list[tuple[str, list[str]]] = [
    ("run ORA on EGFR KRAS BRAF PTEN NRAS", ["EGFR", "KRAS", "BRAF", "PTEN", "NRAS"]),
    (
        "run a pathway analysis on EGFR, KRAS, BRAF, PTEN, NRAS, PIK3CA",
        ["EGFR", "KRAS", "BRAF", "PTEN", "NRAS", "PIK3CA"],
    ),
    (
        "Please run an enrichment on:\nEGFR\nKRAS\nBRAF\nPTEN\nNRAS\nAKT1",
        ["EGFR", "KRAS", "BRAF", "PTEN", "NRAS", "AKT1"],
    ),
    ("Analyse these:\nTP53\nMDM2\nCDKN1A", ["TP53", "MDM2", "CDKN1A"]),
    (
        "Can you analyse these genes?\negfr\nkras\nbraf\npten",
        ["egfr", "kras", "braf", "pten"],
    ),
    ("run pathway analysis: egfr, kras, braf, pten", ["egfr", "kras", "braf", "pten"]),
    (
        "which pathways are my genes in? TP53, MDM2, CDKN1A, BAX",
        ["TP53", "MDM2", "CDKN1A", "BAX"],
    ),
    (
        "What pathways are over-represented in this list: TP53, MDM2, CDKN1A",
        ["TP53", "MDM2", "CDKN1A"],
    ),
    (
        "Here is my gene list, run it through Reactome: TP53 MDM2 CDKN1A BAX",
        ["TP53", "MDM2", "CDKN1A", "BAX"],
    ),
    (
        "TP53, MDM2, CDKN1A, BAX -- run an enrichment please",
        ["TP53", "MDM2", "CDKN1A", "BAX"],
    ),
    ("Find enriched pathways for TP53, MDM2, CDKN1A", ["TP53", "MDM2", "CDKN1A"]),
    ("analyze TP53, MDM2, CDKN1A", ["TP53", "MDM2", "CDKN1A"]),
    (
        "Can you run these through the Reactome analysis tool: TP53, MDM2, BAX",
        ["TP53", "MDM2", "BAX"],
    ),
    (
        "run ORA on my list: IL6 TNF IL1B CXCL8 CCL2",
        ["IL6", "TNF", "IL1B", "CXCL8", "CCL2"],
    ),
    (
        "Please analyse the following proteins: P04637, Q00987, P38936",
        ["P04637", "Q00987", "P38936"],
    ),
    ("do an analysis of these genes: egfr kras braf", ["egfr", "kras", "braf"]),
    (
        "I have a list of DE genes: MYC, CCND1, CDK4, E2F1. Which Reactome pathways are enriched?",
        ["MYC", "CCND1", "CDK4", "E2F1"],
    ),
    ("enrichment for egfr, kras, braf, pten please", ["egfr", "kras", "braf", "pten"]),
    (
        "can you do pathway enrichment on ENSG00000141510 ENSG00000135679",
        ["ENSG00000141510", "ENSG00000135679"],
    ),
    ("map these genes to pathways: TP53 MDM2 CDKN1A", ["TP53", "MDM2", "CDKN1A"]),
    (
        "Run an enrichment on GAPDH, ACTB, TUBB, VIM, KRT8, KRT18",
        ["GAPDH", "ACTB", "TUBB", "VIM", "KRT8", "KRT18"],
    ),
    (
        "run an ORA on CD4 CD8A CD3E CD19 MS4A1 NCAM1 ITGAM",
        ["CD4", "CD8A", "CD3E", "CD19", "MS4A1", "NCAM1", "ITGAM"],
    ),
    (
        "Run an ORA on EGFR, ERBB2, ERBB3, ERBB4, GRB2, SOS1, HRAS",
        ["EGFR", "ERBB2", "ERBB3", "ERBB4", "GRB2", "SOS1", "HRAS"],
    ),
    (
        "Perform a pathway analysis on mouse genes Trp53, Mdm2, Cdkn1a",
        ["Trp53", "Mdm2", "Cdkn1a"],
    ),
    (
        "please run a gene set enrichment on BRCA1 BRCA2 ATM ATR CHEK2",
        ["BRCA1", "BRCA2", "ATM", "ATR", "CHEK2"],
    ),
]

HELD_OUT_REQUESTS: list[tuple[str, list[str]]] = [
    (
        "can we do a gsa analysis in the chat. I want to do it with genes TP53, ERBB2 and RUNX2",
        ["TP53", "ERBB2", "RUNX2"],
    ),
    ("run a pathway analysis on TP53, ERBB2, RUNX2", ["TP53", "ERBB2", "RUNX2"]),
    (
        "Could you run an over-representation analysis for SOX9, RUNX2, SP7, COL1A1?",
        ["SOX9", "RUNX2", "SP7", "COL1A1"],
    ),
    (
        "Please perform an enrichment on these: NFKB1, RELA, IKBKB, CHUK",
        ["NFKB1", "RELA", "IKBKB", "CHUK"],
    ),
    (
        "I'd like to run ORA with STAT1, STAT2, IRF9, ISG15, MX1",
        ["STAT1", "STAT2", "IRF9", "ISG15", "MX1"],
    ),
    (
        "analyse my genes: pik3ca, akt1, mtor, rps6kb1",
        ["pik3ca", "akt1", "mtor", "rps6kb1"],
    ),
    (
        "run GSEA on\nCDK1\nCCNB1\nPLK1\nAURKA\nBUB1",
        ["CDK1", "CCNB1", "PLK1", "AURKA", "BUB1"],
    ),
    ("Do a Reactome enrichment for P04637 and P38398", ["P04637", "P38398"]),
    (
        "find pathways enriched in: SLC2A1, HK2, PFKP, LDHA, PKM",
        ["SLC2A1", "HK2", "PFKP", "LDHA", "PKM"],
    ),
    (
        "which pathways contain these genes? FGF2, FGFR1, FRS2, GRB2",
        ["FGF2", "FGFR1", "FRS2", "GRB2"],
    ),
    ("Run an analysis on insr, irs1, irs2, pik3r1", ["insr", "irs1", "irs2", "pik3r1"]),
    ("perform ORA: VEGFA; KDR; FLT1; NRP1", ["VEGFA", "KDR", "FLT1", "NRP1"]),
    (
        "please analyze CASP3, CASP8, CASP9 and APAF1",
        ["CASP3", "CASP8", "CASP9", "APAF1"],
    ),
    (
        "Can you run an enrichment analysis with my list of genes: Nanog, Pou5f1, Sox2, Klf4",
        ["Nanog", "Pou5f1", "Sox2", "Klf4"],
    ),
    (
        "submit TLR4, MYD88, TRAF6, IRAK4 for pathway analysis",
        ["TLR4", "MYD88", "TRAF6", "IRAK4"],
    ),
]

REVIEW_QUESTIONS: list[str] = [
    "Can you explain the enrichment of TP53 targets in apoptosis pathways?",
    "Is BRCA1 enriched in DNA repair pathways compared to BRCA2?",
    "Please explain how EGFR and KRAS signalling interact",
    "Can you tell me which pathways are enriched in my GSEA results for MYC and E2F targets?",
    "What does pathway analysis tell us about TP53 and MDM2?",
    "Why is my enrichment analysis showing ERBB2 and GRB7 together?",
    "I ran an ORA and got TP53 and CDKN1A at the top, what does that mean?",
    "Could you describe the role of SMAD2 and SMAD3 in TGF-beta signalling?",
    "Please summarise what is known about BRCA1 and BRCA2",
    "Can you do a literature summary on IL6 and STAT3?",
    "In a gene set analysis, why would HLA-A and B2M both show up?",
    "How does over-representation analysis handle genes like TP53 that are in many pathways?",
    "What pathways involve both PTEN and AKT1?",
    "Can I use UniProt IDs like P04637 and P38398 in the analysis tool?",
    "Is GSEA better than ORA for a list like CD4, CD8A, CD3E?",
    "Could you explain why IFNG and TNF are enriched in immune pathways?",
    "Please help me understand the interaction between MTOR and RPTOR",
    "What does a p<0.05 FDR mean for pathway analysis of NOTCH1 targets?",
    "Can you show me the reactions in which CDK1 and CCNB1 take part?",
    "I want to understand the enrichment of JAK2 and STAT5A in hematopoiesis",
    "Run me through how pathway analysis works for genes like APOE and APP",
    "Can you list the pathways where VEGFA and KDR are found?",
    "Would like to know if ESR1 and PGR are enriched in breast cancer pathways",
    "Help me interpret this: WNT signalling enriched, CTNNB1 and APC found",
    "Can you do a comparison of MAPK1 and MAPK3 functions?",
    "Please describe the reaction where ATP is hydrolysed by ABCB1",
    "How is NF-kB activated by TNF and IL1B?",
    "Which genes in the TCA cycle, like CS and IDH2, are enriched in my data?",
    "What is the enrichment score for TP53 in the Reactome GSA output?",
    "Could we discuss pathway analysis results where SOX2 and POU5F1 appear?",
    "I did an enrichment with DAVID for FOXO1 and FOXO3, can you compare with Reactome?",
    "Can you explain the difference between ORA and GSEA using BRCA1 and ATM as examples?",
    "Please tell me whether KRAS G12D is enriched in pancreatic cancer",
    "Can you check if my pathway analysis with PD-1 and CTLA4 is correct?",
    "Could you help me with an analysis of how TP53 and MDM2 regulate each other?",
    "Is there an enrichment of ERBB2 amplification in HER2+ breast cancer?",
    "Can you run through the steps of the insulin pathway: INS, INSR, IRS1?",
    "Do HIF1A and VHL appear together in any enriched pathway?",
    "What are the downstream effects of mTORC1 and mTORC2 in pathway analysis?",
    "Please explain why GAPDH and ACTB are used as housekeeping genes in enrichment",
    "Can you describe the ORA method? I have genes like TP53.",
    "How many genes are in the pathway enriched for BCL2 and BAX?",
    "Can we talk about the 2x enrichment of CD19 and MS4A1 in B cells?",
    "What would pathway analysis show for a knockout of IL2 and IL2RA?",
]

HELD_OUT_QUESTIONS: list[str] = [
    "Which pathways is TP53 in?",
    "Can you explain what an ORA does with genes like MYC and MAX?",
    "Why does my enrichment show HLA-A and HLA-B at the top?",
    "How are RUNX2 and SP7 related in osteoblast differentiation?",
    "Is ERBB2 over-represented in breast cancer compared to ERBB3?",
    "What is the role of NOTCH1 and JAG1 in development?",
    "Tell me about BRCA1 and BRCA2 in homologous recombination",
    "Can you summarise the MAPK cascade with BRAF and MEK1?",
    "I ran GSEA and CDK1 and PLK1 came up. Does that make sense?",
    "Please give me the reactions that involve ATM and CHEK2",
    "Could you find the literature on TP53 and MDM2?",
    "What does FDR mean in my analysis with EGFR and KRAS?",
    "Does CTNNB1 bind APC in the destruction complex?",
    "Show me the diagram for the pathway with SMAD2 and SMAD4",
    "What analysis can I do with DESeq2 output for 3 conditions?",
    "run the pathway browser for me",
    "Are IL6 and STAT3 enriched in the JAK-STAT pathway?",
    "Is there a pathway that links GLUT1 and HK2?",
    "Which of these involve TP53?",
    "Can you find pathways where both INS and INSR are present?",
]


SECOND_REVIEW_REQUESTS: list[tuple[str, list[str]]] = [
    ("run ORA on\nTP53\tMDM2\tCDKN1A\tBAX", ["TP53", "MDM2", "CDKN1A", "BAX"]),
    (
        "Enrichment please:\nGene\nTP53\nMDM2\nCDKN1A\nBAX",
        ["TP53", "MDM2", "CDKN1A", "BAX"],
    ),
    ("analyse these genes:\n1. TP53\n2. MDM2\n3. CDKN1A", ["TP53", "MDM2", "CDKN1A"]),
    ("please run an enrichment on:\n- EGFR\n- KRAS\n- BRAF", ["EGFR", "KRAS", "BRAF"]),
    ("perform pathway analysis on:\n* EGFR\n* KRAS\n* BRAF", ["EGFR", "KRAS", "BRAF"]),
    ("run ORA on TP53, P38398, ENSG00000135679", ["TP53", "P38398", "ENSG00000135679"]),
    (
        "run an enrichment on TP53, MDM2, CDKN1A. These came from my knockdown screen.",
        ["TP53", "MDM2", "CDKN1A"],
    ),
    (
        "Run a pathway analysis on these: TP53, MDM2, BAX, thanks!",
        ["TP53", "MDM2", "BAX"],
    ),
    (
        "do an ORA with the following genes\nSTAT1\nSTAT2\nIRF9\n\nThey are interferon genes.",
        ["STAT1", "STAT2", "IRF9"],
    ),
    ("run enrichment on MYC/MAX/MXD1", ["MYC", "MAX", "MXD1"]),
    ("Run ORA on: TP53 | MDM2 | CDKN1A", ["TP53", "MDM2", "CDKN1A"]),
    (
        "Please do an enrichment analysis on my gene list (TP53, MDM2, CDKN1A)",
        ["TP53", "MDM2", "CDKN1A"],
    ),
    ('run ora on "TP53", "MDM2", "CDKN1A"', ["TP53", "MDM2", "CDKN1A"]),
    (
        "Run an enrichment for HLA-A, HLA-B, B2M, TAP1",
        ["HLA-A", "HLA-B", "B2M", "TAP1"],
    ),
    (
        "Can you run an ORA on these mouse genes: Trp53, Mdm2, Cdkn1a, Bax",
        ["Trp53", "Mdm2", "Cdkn1a", "Bax"],
    ),
    (
        "perform enrichment on CD8A, GZMB, PRF1, IFNG, and NKG7",
        ["CD8A", "GZMB", "PRF1", "IFNG", "NKG7"],
    ),
    (
        "Run a Reactome analysis on the list below\n\nIL6\nIL1B\nTNF\nCXCL8",
        ["IL6", "IL1B", "TNF", "CXCL8"],
    ),
    ("find enriched pathways for: sox9, runx2, sp7", ["sox9", "runx2", "sp7"]),
    (
        "run GSEA on TP53, MDM2, CDKN1A, BAX, BBC3, PMAIP1, FAS, TNFRSF10B, GADD45A, SESN1, RRM2B, ZMAT3",
        [
            "TP53",
            "MDM2",
            "CDKN1A",
            "BAX",
            "BBC3",
            "PMAIP1",
            "FAS",
            "TNFRSF10B",
            "GADD45A",
            "SESN1",
            "RRM2B",
            "ZMAT3",
        ],
    ),
    ("run an ORA on P04637-2, Q00987, O15350", ["P04637-2", "Q00987", "O15350"]),
    (
        "Please run enrichment on ATF4 DDIT3 XBP1 ERN1 EIF2AK3",
        ["ATF4", "DDIT3", "XBP1", "ERN1", "EIF2AK3"],
    ),
    (
        "analyse: ENSG00000141510.18, ENSG00000135679.25",
        ["ENSG00000141510", "ENSG00000135679"],
    ),
    (
        "run ORA with genes GATA1, TAL1, KLF1, LMO2 from my erythroid dataset",
        ["GATA1", "TAL1", "KLF1", "LMO2"],
    ),
    (
        "Do an over-representation analysis for PSEN1, APP, APOE, MAPT, TREM2, CLU",
        ["PSEN1", "APP", "APOE", "MAPT", "TREM2", "CLU"],
    ),
]

SECOND_REVIEW_QUESTIONS: list[str] = [
    "Can EGFR and ERBB2 form heterodimers in Reactome?",
    "Are there any pathways where both TP53 and MYC act?",
    "Please list the reactions that SMAD3 and SMAD4 participate in",
    "I'd like the Reactome pathways for CDK4 and CDK6",
    "Which Reactome pathway has the most overlap with BRCA1, BRCA2 and PALB2?",
    "Find me papers about KRAS and NRAS in colorectal cancer",
    "Does Reactome have an analysis of IL6 versus IL10 signalling?",
    "Show me where PIK3CA, AKT1 and MTOR sit in the PI3K pathway",
    "Map out the interactions between NOTCH1 and HES1 for me",
    "Run a search for pathways containing JAK1 and JAK2",
    "Which pathways are shared by STAT1 and STAT3?",
    "In my enrichment analysis, CXCL8 and CCL2 were top hits. Anything to worry about?",
    "Find the reactions where MDM2 ubiquitinates TP53",
    "Can you find the complexes that contain CDK1 and CCNB1?",
    "I performed an ORA with 200 genes, of which TP53 and ATM were most significant. Next steps?",
    "Submit a question: are ESR1 and FOXA1 co-regulated?",
    "Please run a check on whether BAX and BAK1 are in apoptosis",
    "Map the protein P04637 and P38398 to their Reactome names",
    "Find pathways related to insulin, e.g. INS, INSR",
    "Which pathways are regulated by miRNAs targeting PTEN and TP53?",
    "Is the GSEA leading edge with ACTB and GAPDH normal?",
    "Execute a query for interactors of VEGFA and KDR",
    "Can I analyse a gene list of 500 genes including TP53 and EGFR?",
    "Can you find what pathways CD4 and CD8A are enriched for in T cells, per the literature?",
    "Which pathways are my DE genes in? I have not decided yet, maybe TP53 and MYC.",
    "I want to run ORA later; first, what are HIF1A and EPAS1?",
    "Find enrichment papers on BRCA1 and BRCA2 please",
    "Run me a summary: TP53, MDM2",
    "Do an analysis of the TP53-MDM2 feedback loop",
    "Please analyse the role of TNF and IL1B in inflammation",
    "Analyse the relationship between APC and CTNNB1",
    "Which pathways do SOX2 and NANOG regulate?",
    "Find the enrichment map for my results; genes include FOS and JUN",
    "Can you perform a literature search on GSEA results with MYC and E2F1?",
    "Carry out a comparison between AKT1 and AKT2 please",
    "Which pathways involve TGFB1 and SMAD7 negatively?",
    "Map EGFR mutations L858R and T790M to pathways",
    "Run through the analysis steps for MAPK1 and MAPK3",
    "Find the pathways on chromosome 17 with TP53 and BRCA1",
    "Is ORA appropriate for TP53, MDM2 and CDKN1A or should I use GSEA?",
]

HELD_OUT_3_REQUESTS: list[tuple[str, list[str]]] = [
    (
        "Hi! Could you run a pathway enrichment on NOTCH1, NOTCH2, JAG1, DLL4, HES1?",
        ["NOTCH1", "NOTCH2", "JAG1", "DLL4", "HES1"],
    ),
    (
        "run ORA:\nABCA1\nABCG1\nAPOA1\nLDLR\nPCSK9",
        ["ABCA1", "ABCG1", "APOA1", "LDLR", "PCSK9"],
    ),
    (
        "please analyse this list: Myod1, Myog, Myf5, Des",
        ["Myod1", "Myog", "Myf5", "Des"],
    ),
    (
        "Can you do an enrichment analysis on CXCR4, CXCL12, ACKR3?",
        ["CXCR4", "CXCL12", "ACKR3"],
    ),
    ("run pathway analysis on P01308, P06213, P35568", ["P01308", "P06213", "P35568"]),
    (
        "Perform ORA on BCL2, BCL2L1, MCL1, BAX, BAK1, BID. Thanks in advance",
        ["BCL2", "BCL2L1", "MCL1", "BAX", "BAK1", "BID"],
    ),
    ("find enriched pathways for\nhk1\nhk2\ngck\npfkl", ["hk1", "hk2", "gck", "pfkl"]),
    (
        "run an enrichment on SREBF1; SREBF2; INSIG1; SCAP",
        ["SREBF1", "SREBF2", "INSIG1", "SCAP"],
    ),
    (
        "Could we run a GSA with TNFRSF1A, TRADD, RIPK1 and TRAF2?",
        ["TNFRSF1A", "TRADD", "RIPK1", "TRAF2"],
    ),
    (
        "analyse the following:\nCOL1A1\nCOL1A2\nCOL3A1\nFN1\n\nThese are fibrosis markers",
        ["COL1A1", "COL1A2", "COL3A1", "FN1"],
    ),
    (
        "do a pathway analysis with ATG5, ATG7, BECN1, MAP1LC3B, SQSTM1",
        ["ATG5", "ATG7", "BECN1", "MAP1LC3B", "SQSTM1"],
    ),
]

HELD_OUT_3_QUESTIONS: list[str] = [
    # Was a known limit; fixed by the two-genes-in-a-question rule.
    "Is my list of TP53, MDM2 targets enriched for apoptosis? I have not run anything yet, just asking",
    "Could you explain whether SOX9 and RUNX2 act together?",
    "Which Reactome pathway includes both ATG5 and ATG7?",
    "What happens to BCL2 and BAX during apoptosis?",
    "Can you run a quick check: is TP53 a tumour suppressor?",
    "Do FOXP3 and IL2RA mark regulatory T cells?",
    "Please find reactions catalysed by HK1 and HK2",
    "Give me an overview of enrichment analysis for genes like CXCR4 and CXCL12",
    "How do NOTCH1 and JAG1 signal?",
    "Show me the pathway diagram containing LDLR and PCSK9",
    "Which pathways would you expect to be enriched for COL1A1 and FN1?",
    "Help me write a methods section for my ORA of TP53 and MDM2",
    "What's the Reactome ID for SREBF1 and SREBF2?",
    "Run a literature search on MYOD1 and MYOG",
    "Can you describe what TNFRSF1A and TRADD do in NF-kB activation?",
]

#: What a request followed by a trailing sentence must still read as.
TRAILING_BASE = "run an enrichment on TP53, MDM2, CDKN1A"
TRAILING: list[str] = [
    " from my screen",
    " using default settings",
    " thanks",
    " cheers",
    " asap",
    " today",
    " vs background",
    " against the human genome",
    "\n\nCheers, Anna",
    "\nThanks",
    " please and thank you",
    " (human)",
    " - human",
    " if possible",
    " when you can",
    " at FDR 0.05",
    " using Reactome",
    " by FDR",
    " sorted by pvalue",
    " but exclude disease pathways",
    " etc",
    " genes",
    "\n\nBest\nJohn",
]

THIRD_REVIEW_REQUESTS: list[tuple[str, list[str]]] = [
    (
        "Could you run an over-representation analysis for SOX2, POU5F1, NANOG, KLF4 and LIN28A?",
        ["SOX2", "POU5F1", "NANOG", "KLF4", "LIN28A"],
    ),
    (
        "run enrichment on these DE genes:\nIL6\nCXCL8\nCCL2\nTNF\nIL1B\nPTGS2",
        ["IL6", "CXCL8", "CCL2", "TNF", "IL1B", "PTGS2"],
    ),
    ("Perform ORA with P04637, Q00987, P38936", ["P04637", "Q00987", "P38936"]),
    (
        "please do a reactome analysis of ENSG00000141510, ENSG00000135679, ENSG00000124762",
        ["ENSG00000141510", "ENSG00000135679", "ENSG00000124762"],
    ),
    (
        "map BRCA1 BRCA2 PALB2 RAD51C RAD51D to pathways",
        ["BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"],
    ),
    (
        "Find the enriched pathways for my hits: Atg5, Atg7, Becn1, Map1lc3b, Sqstm1",
        ["Atg5", "Atg7", "Becn1", "Map1lc3b", "Sqstm1"],
    ),
    (
        "hi! can you run a pathway enrichment on CD3E, CD4, CD8A, GZMB, PRF1, IFNG thanks",
        ["CD3E", "CD4", "CD8A", "GZMB", "PRF1", "IFNG"],
    ),
    (
        "Run ORA:\nHIF1A, VEGFA, EPAS1, LDHA, PDK1, SLC2A1",
        ["HIF1A", "VEGFA", "EPAS1", "LDHA", "PDK1", "SLC2A1"],
    ),
    (
        "I'd like you to analyse NOTCH1, HES1, HEY1, JAG1, DLL4 for pathway enrichment",
        ["NOTCH1", "HES1", "HEY1", "JAG1", "DLL4"],
    ),
    (
        "Submit this list to Reactome analysis: MLH1; MSH2; MSH6; PMS2",
        ["MLH1", "MSH2", "MSH6", "PMS2"],
    ),
    (
        "do an enrichment analysis on the following genes: ATM, ATR, CHEK1, CHEK2, WEE1",
        ["ATM", "ATR", "CHEK1", "CHEK2", "WEE1"],
    ),
    (
        "run gsa with KEAP1, NFE2L2, NQO1, HMOX1, GCLM",
        ["KEAP1", "NFE2L2", "NQO1", "HMOX1", "GCLM"],
    ),
    (
        "Which pathways are enriched in STAT1 STAT2 IRF9 ISG15 MX1 OAS1?",
        ["STAT1", "STAT2", "IRF9", "ISG15", "MX1", "OAS1"],
    ),
    (
        "Run enrichment for:\n1. PINK1\n2. PRKN\n3. LRRK2\n4. SNCA\n5. PARK7",
        ["PINK1", "PRKN", "LRRK2", "SNCA", "PARK7"],
    ),
    (
        "analyze my genes - CTNNB1, APC, AXIN2, LGR5, TCF7L2 - using reactome",
        ["CTNNB1", "APC", "AXIN2", "LGR5", "TCF7L2"],
    ),
    (
        "please perform over representation analysis: Tp53, Mdm2, Cdkn1a, Bax, Pmaip1",
        ["Tp53", "Mdm2", "Cdkn1a", "Bax", "Pmaip1"],
    ),
    (
        "run an ORA on SMAD2/SMAD3/SMAD4/TGFBR1/TGFBR2",
        ["SMAD2", "SMAD3", "SMAD4", "TGFBR1", "TGFBR2"],
    ),
    (
        "Can you do pathway analysis on my list? MYOD1, MYOG, MEF2C, DES, ACTA1",
        ["MYOD1", "MYOG", "MEF2C", "DES", "ACTA1"],
    ),
    (
        "execute an enrichment on EZH2 SUZ12 EED RBBP4",
        ["EZH2", "SUZ12", "EED", "RBBP4"],
    ),
    (
        "Analyse for pathway enrichment:\nGAPDH\nPGK1\nENO1\nPKM\nALDOA\n\nThanks,\nMaria",
        ["GAPDH", "PGK1", "ENO1", "PKM", "ALDOA"],
    ),
]

THIRD_REVIEW_QUESTIONS: list[str] = [
    "Is KRAS or NRAS more frequently mutated in colorectal cancer pathways?",
    "Can you find pathways where both PTEN and PIK3CA act?",
    "What happens downstream of EGFR and ERBB2 activation?",
    "Which pathways does TP53 participate in, and does MDM2 share any?",
    "Run me through the steps of mismatch repair involving MLH1 and MSH2",
    "Has anyone performed an enrichment analysis with BRCA1 and BRCA2 knockouts in Reactome?",
    "I ran ORA on my list and got Signaling by NOTCH at the top with NOTCH1, JAG1 - is that plausible?",
    "Do SOX2, POU5F1 appear together in any Reactome pathway?",
    "After I run the analysis on IL6, TNF, should I use the projection to human?",
    "Which pathways are shown when I analyze CD19, MS4A1 in the Pathway Browser - where do I click?",
    "What's the Reactome ID for the pathway containing ATM, CHEK2?",
    "My enrichment for HIF1A, VEGFA came back empty. What went wrong?",
    "Map of the interactions between KEAP1 and NFE2L2 please",
    "Could analysis of STAT3, JAK2 phosphorylation be done in Reactome?",
    "Is GAPDH a good housekeeping gene for ORA background, along with ACTB?",
    "In which pathways would BCL2, BAX, BAK1 be found?",
    "Please find me literature-backed pathways for FOXO1, FOXO3",
    "Should I run an ORA with SMAD4, TGFBR2 or wait until I have more genes?",
    "Why doesn't my ORA list CDK4, CDK6 in cell cycle?",
    "Run the numbers for me: are MYC, MAX, MXD1 all in the same Reactome pathway?",
]

HELD_OUT_4_REQUESTS: list[tuple[str, list[str]]] = [
    (
        "Run a pathway enrichment on KEAP1, NFE2L2, HMOX1, NQO1 please",
        ["KEAP1", "NFE2L2", "HMOX1", "NQO1"],
    ),
    (
        "perform ORA with the following:\nPINK1\nPRKN\nPARK7\nLRRK2\nSNCA",
        ["PINK1", "PRKN", "PARK7", "LRRK2", "SNCA"],
    ),
    (
        "Could you do an over-representation analysis of GLS, GLUD1, GOT2, SLC1A5?",
        ["GLS", "GLUD1", "GOT2", "SLC1A5"],
    ),
    ("analyse these: cd274, pdcd1, ctla4, lag3", ["cd274", "pdcd1", "ctla4", "lag3"]),
    (
        "Enrichment analysis on SIRT1, SIRT3, PPARGC1A, FOXO3 - can you run it?",
        ["SIRT1", "SIRT3", "PPARGC1A", "FOXO3"],
    ),
    ("run ORA on Q16539, P45983, P53779", ["Q16539", "P45983", "P53779"]),
    (
        "Can we run an enrichment for ACE2, TMPRSS2, FURIN and CTSL",
        ["ACE2", "TMPRSS2", "FURIN", "CTSL"],
    ),
    (
        "please do a GSEA with RB1, E2F1, CDK2, CCNE1, CDKN1B",
        ["RB1", "E2F1", "CDK2", "CCNE1", "CDKN1B"],
    ),
    (
        "run pathway analysis on WNT3A FZD7 LRP6 DVL2 AXIN1",
        ["WNT3A", "FZD7", "LRP6", "DVL2", "AXIN1"],
    ),
]

HELD_OUT_4_QUESTIONS: list[str] = [
    "What does KEAP1 do to NFE2L2 under oxidative stress?",
    "Is PINK1, PRKN signalling part of mitophagy in Reactome?",
    "Can you tell me if GLS and GLUD1 are in glutamine metabolism?",
    "Why are CD274 and PDCD1 targets for immunotherapy?",
    "Which pathway would SIRT1, SIRT3, FOXO3 all belong to?",
    "Run a search for the ACE2 entry please",
    "Can I use Ensembl IDs like ENSG00000130234 and ENSG00000184012 for analysis?",
    "What is the difference between RB1 and CDKN1B?",
    "Show me the reactions for PAX6 and SOX1 in neural development",
    "How do WNT3A and FZD7 activate beta-catenin?",
    "Where is LRRK2 located in the cell?",
    "I ran an enrichment of SNCA, LRRK2, PARK7 last week - is the result still valid after the release?",
]

#: Current behaviour that is wrong, pinned so a change is noticed.
KNOWN_LIMITS: list[tuple[str, list[str] | None]] = [
    (
        "I'd like an over-representation analysis on these genes - FOXP3, IL2RA, CTLA4, IKZF2",
        None,
    ),
    (
        "gene\tlog2FC\tpadj\nTP53\t2.1\t0.001\nMDM2\t1.5\t0.01\nCDKN1A\t3.2\t0.0001\nplease run an enrichment analysis",
        None,
    ),
    ("find pathways for my genes: Pax6, Sox1, Nes, Otx2", None),
]
