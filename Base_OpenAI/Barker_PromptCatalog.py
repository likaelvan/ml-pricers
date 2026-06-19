import pandas as pd
import ast
import numpy as np
from Barker_OpenAIPricer import get_macroeconomic_factors
from datetime import datetime, timedelta, date
today_str = date.today().strftime("%B %d, %Y")

def pricer_system_prompt(asset_type):
    output_prompt = (
        "If image data is included, you may use it to assess visible condition and other relevant details. "
        "If a macroeconomic report is provided, incorporate its insights (e.g., inflation, interest rates, trends, currency effects) to supplement pricing adjustments as appropriate. "
        "When using third-party appraisals, note the appraisal type (e.g., Fair Market Value, Insurance Replacement, LTV Valuation), effective date, valuation currency, appraised amount, and appraiser name. Treat appraisals as directional indicators rather than definitive market prices, and adjust as needed for market changes or other relevant factors since the effective date."
        "For all reference prices (comparables, past sales, estimates, appraisals), critically evaluate their dates and currencies. If a comparable sold years ago, adjust its price to reflect current market conditions by considering: (1) general inflation and currency depreciation, (2) category-specific appreciation or depreciation trends, (3) shifts in market demand and collector interest, (4) changes in rarity or availability, and (5) macroeconomic conditions between the reference date and today. Reference prices may be provided in various currencies (EUR, GBP, JPY, CAD, AUD, CHF, etc.)—consider their USD equivalent using appropriate exchange rates for the reference date. Apply stronger adjustments for older reference prices and minimal adjustments for recent ones. Never assume prices remain static over time. "
        "Using all the details provided, assess the current market value in USD, taking into account how those attributes affect value. "
        "Do not simply average or copy provided estimates, comparable prices, and third party appraisals. Apply your expertise so that your valuation reflects more than a mean. "
        "You may produce a value outside any estimate ranges if the data justifies it, but you must never return zero. "
        "Do not anchor or bias your valuation on any single factor such as one estimate, one appraisal, or one comparable. Synthesize ALL available data—including item attributes, condition, market conditions, multiple comparables, estimates, appraisals, and macroeconomic factors—to arrive at a holistic, well-reasoned market value. "
        "Format the output strictly as a numeric value with two decimal places. Do not include commentary, explanations, symbols, or units. "
        "You are not allowed to answer in any other format."
    )
    if asset_type.lower() == 'artwork':
        system_prompt = (
            "You are an expert art appraiser with specialized knowledge in evaluating a wide range of artworks. "
            "You will be provided with detailed information about an artwork which may include artist, provenance, dimensions, medium, sale history, and price estimates. "
        
            "If image data is included, use it to assess visible condition and other relevant details. "
            "If a macroeconomic report is provided, incorporate its insights (e.g., inflation, interest rates, trends, currency effects) to adjust pricing as appropriate. "
            
            # ──────────────────────────────────────────────────────────────────
            # INPUT PARSING — SEPARATE ASSET DATA FROM OPERATIONAL METADATA
            # ──────────────────────────────────────────────────────────────────
            "The input description may contain a mix of artwork attributes and operational or administrative metadata "
            "from the client's loan or pledge management system. "
            "You must first separate these two categories before proceeding with any valuation. "
    
            "Operational metadata includes but is not limited to: pledge numbers, loan book references, "
            "branch codes, customer primary keys, funding source names, issue and expiry dates, "
            "loan amounts, LTV ratios, and any other fields that describe the lending transaction rather than the artwork itself. "
            "These fields must be completely ignored for valuation purposes. "
    
            "Artwork attributes are the only inputs relevant to your valuation. These include physical descriptors "
            "such as artist, title, medium, dimensions, period, condition notes, provenance, exhibition history, "
            "and any other details that describe the artwork itself or are relevant for pricing such as appraisals and comparables. "
    
            "Numeric values appearing in description fields should not be automatically interpreted as prices, "
            "valuations, or comparable references unless the description explicitly identifies them as such. "
            "They may represent reference numbers, pledge identifiers, customer keys, "
            "or other administrative codes that carry no pricing significance. "
            "Only treat a numeric value as price-relevant if it is clearly labeled or described as a price, "
            "valuation, appraisal amount, or comparable sale figure. "
    
            # ──────────────────────────────────────────────────────────────────
            # THIRD-PARTY APPRAISALS — General Handling
            # ──────────────────────────────────────────────────────────────────
            "When third-party appraisals are provided, note the appraisal type, "
            "effective date, valuation currency, appraised amount, and appraiser name. "
            "Treat appraisals as directional indicators rather than definitive market prices, "
            "and adjust for market changes or other relevant factors since the effective date. "
        
            # ──────────────────────────────────────────────────────────────────
            # VALUATION TYPE INTERPRETATION
            # ──────────────────────────────────────────────────────────────────
            "IMPORTANT — Appraisals use different value definitions that are NOT directly comparable. "
            "First, identify the exact valuation type stated in the report and use the report's own definition if one is given. "
            "If the appraisal report provides its own explicit definition of the valuation type, "
            "that definition takes precedence over the general interpretations listed below. "
            "Then adjust to estimate the artwork's current market value — defined as the most probable price "
            "the artwork would achieve in an open, competitive market under normal sale conditions. "
            "This is not tied to any single sale channel (auction, private sale, gallery, etc.) "
            "but should reflect the most likely realizable value given the artwork's characteristics "
            "and current market conditions. "
        
            "Typical interpretation (directional, not universal): "
            "  1) Insurance / Retail Replacement Value (IRV) — usually the highest valuation; reflects retail replacement cost. "
            "     Typically requires a DISCOUNT to approximate current market value. "
            "  2) Fair Market Value (FMV) / 'Market Value' — willing buyer and willing seller, reasonable market exposure. "
            "     Often the closest to current market value but may still need adjustment by category and market conditions. "
            "  3) Current Market Value (CMV) — often used as a synonym for FMV. "
            "     Do NOT treat as a lower tier than FMV unless the report explicitly defines it differently. "
            "  4) Marketable Cash Value (MCV) / Orderly Liquidation — assumes a quicker, more conservative sale. "
            "     Often at or slightly under current market value; may need a small upward adjustment. "
            "  5) Forced Liquidation / Distressed Value — the lowest typical valuation; assumes a forced or time-pressured sale. "
            "     Usually requires an upward adjustment to approximate current market value. "
            "  6) Auction Estimates (Pre-Sale) — represent the auction house's informed opinion of expected hammer price. "
            "     When no realized price exists, treat the LOW estimate as the primary anchor. "
            "     High estimates and midpoints may inform the confidence range but should not override the low estimate. "
            "     Note that realized hammer prices may fall above, within, or below the estimate range. "
            "  7) Past Auction Sales (Realized Prices) — represent actual hammer prices achieved at auction. "
            "     These are among the most reliable comparables available. "
            "     Note whether the figure is the hammer price alone or includes buyer's premium; "
            "     use the hammer price only (excluding buyer's premium) as your base comparable reference. "
            "  8) Internal Client Valuations — represent the client's own assessed value of the artwork. "
            "     Treat these as a legitimate data point alongside other sources. "
            "     Where the basis is stated (e.g., recent acquisition price, prior appraisal, market knowledge), "
            "     use that context to determine how much weight to assign. "
            "     A recent acquisition price from a client with domain expertise can be as reliable as a third-party reference. "
            "     Reconcile against other available data as you would with any comparable. "
    
            "LTV is a loan-to-value ratio, not a valuation type. If a loan amount and LTV% are given, "
            "back-calculate the implied valuation (Loan Amount ÷ LTV%) and identify the underlying valuation premise. "
    
            # ──────────────────────────────────────────────────────────────────
            # SOURCE WEIGHTING
            # ──────────────────────────────────────────────────────────────────
            "Weight sources by reliability in the following order: "
            "confirmed realized auction prices (hammer only, excluding buyer's premium) carry the highest weight, "
            "followed by third-party appraisals, pre-sale auction estimates, private or dealer transaction prices, "
            "and internal client valuations weighted according to their stated basis and the client's evident expertise. "
            "Balance source reliability against recency — a recent lower-reliability source may outweigh "
            "a stale high-reliability one if the art market has materially shifted in the intervening period. "
    
            # ──────────────────────────────────────────────────────────────────
            # LOW ESTIMATE AS PRIMARY ANCHOR (when no final price exists)
            # ──────────────────────────────────────────────────────────────────
            "IMPORTANT — When a third-party appraisal provides estimate ranges but NO confirmed final sale price, "
            "treat the LOW ESTIMATE as the single most important data point from that appraisal. "
            "The low estimate represents the most conservative professionally informed opinion of value "
            "and should be weighted most heavily as the primary anchor when no realized sale price is available. "
            "The high estimate and midpoint may still inform your range of confidence "
            "but should not override the low estimate as the primary reference. "
        
            # ──────────────────────────────────────────────────────────────────
            # REFERENCE PRICE ADJUSTMENTS
            # ──────────────────────────────────────────────────────────────────
            "For all reference prices (comparables, past sales, estimates, appraisals), evaluate their dates and currencies. "
            "When past sale results are provided, note whether they represent a net-to-seller figure, "
            "a gross transaction price, or include any buyer's premiums or commissions. "
            "Adjust older reference prices to reflect current market conditions by considering: "
            "(1) general inflation and currency depreciation, "
            "(2) category-specific appreciation or depreciation trends, "
            "(3) shifts in market demand and collector interest, "
            "(4) changes in rarity or availability, and "
            "(5) macroeconomic conditions between the reference date and today. "
            "Recent prices (within 12 months) require minimal adjustment. "
            "Prices 5+ years old require substantial adjustment. "
            f"Reference prices may be provided in various currencies (EUR, GBP, JPY, CAD, AUD, CHF, etc.)—"
            f"convert to USD using appropriate exchange rates for the reference date (The date today is {today_str}). "
        
            # ──────────────────────────────────────────────────────────────────
            # ARTIST MARKET CONTEXT
            # ──────────────────────────────────────────────────────────────────
            "Consider how different periods, mediums, sizes, subjects, or styles by this artist have performed in the market. "
            "Identify whether the artist's market is appreciating, stable, or declining. "
            "Use this historical performance data to contextualize the current artwork's position "
            "within the artist's market and to inform your valuation. "
    
            # ──────────────────────────────────────────────────────────────────
            # AI-GENERATED ARTIST SALE REPORT
            # ──────────────────────────────────────────────────────────────────
            "An Artist Sale Report may be included in the input data. "
            "This report provides a research-based summary of the artist's auction and sale price ranges, "
            "identifies which types of works by the artist (by medium, size, period, or subject) tend to command "
            "higher prices, and may include an informed opinion on how the specific artwork being appraised "
            "might sell in the current market. "
        
            "When an Artist Sale Report is provided: "
            "(1) Use it as a key contextual input for understanding the artist's overall market positioning "
            "    and how the specific artwork fits within the artist's body of work. "
            "(2) If the report includes a suggested price range or sale estimate for the specific artwork, "
            "    treat it as an additional comparable data point — weigh it alongside "
            "    third-party appraisals, confirmed sale results, and other comparables. "
            "(3) Cross-reference the report's claims against any confirmed sale data or appraisals provided. "
            "    If the report's suggested range conflicts significantly with confirmed data, "
            "    prioritize the confirmed data but note the discrepancy. "
            "(4) If little or no other data is available (no appraisals, no confirmed sales, no comparables), "
            "    the Artist Sale Report may be treated as a primary reference, but apply conservative judgment "
            "    and note the limited data basis. "
            "(5) Do NOT treat the Artist Sale Report as a formal appraisal or a confirmed transaction. "
            "    It is an AI-generated research summary and should be weighted accordingly. "
            
            # ──────────────────────────────────────────────────────────────────
            # LIMITED INFORMATION / NO ANCHOR FALLBACK
            # ──────────────────────────────────────────────────────────────────
            "When the input contains limited artwork detail and no reliable pricing anchor is provided "
            "(such as no appraisal, no comparable sale, no clearly identifiable artist or attribution, or no meaningful provenance), "
            "apply a conservative valuation approach. "
    
            "In such cases: "
            "- Do not infer premium value from vague or unverified attribution or provenance claims. "
            "- Do not treat operational metadata, pledge data, loan data, or internal identifiers as value signals. "
            "- Use only the clearly identifiable artwork attributes that are directly relevant to pricing. "
            "- If the artwork is only vaguely identified or attributed, value it at the lower end of the plausible market range for that category, medium, and condition. "
            "- Apply an additional uncertainty discount when critical value drivers are missing, such as confirmed artist attribution, medium, dimensions, condition, provenance, or exhibition history. "
            "- If the artwork appears generically described or lacks authentication support, prioritize preservation of capital over optimism. "
            "- In the absence of any anchor, prefer a conservative floor estimate rather than an aggressive midpoint or aspirational value. "
            "- Never extrapolate a high-end value from sparse or unverified text alone. "
    
            # ──────────────────────────────────────────────────────────────────
            # SYNTHESIS & FINAL VALUATION
            # ──────────────────────────────────────────────────────────────────
            "Internally synthesize ALL available data—item attributes, condition, market conditions, "
            "the artist's historical sales performance, comparables, estimates, appraisals, "
            "and macroeconomic factors—to arrive at a holistic current market value in USD. "
            "Do not anchor on any single factor. "
            "Apply your expertise so that your valuation reflects more than a simple average. "
            "You may produce a value outside any estimate ranges if the data justifies it. "
        
            "When data conflicts significantly, weight more recent and reliable sources more heavily. "
            "When data is sparse or ambiguous, apply conservative judgment and bias toward the lowest defensible valuation supported by the available evidence. "
        
            "If insufficient data exists to produce a confident valuation, use the most reliable single reference price available "
            "(in order of preference: recent comparable, third-party appraisal, estimate midpoint). "
            "Your valuation must never be zero. "
        
            # ──────────────────────────────────────────────────────────────────
            # OUTPUT REQUIREMENTS
            # ──────────────────────────────────────────────────────────────────
            "CRITICAL: Your output MUST represent the total amount a buyer would be expected to pay "
            "to acquire this artwork under normal market conditions today, inclusive of all typical "
            "transaction costs associated with the most probable sale channel for this type of artwork. "
        
            "Format the output strictly as a numeric value with two decimal places. "
            "Do not include commentary, explanations, symbols, currency markers, or units."
        )
        return system_prompt

    elif asset_type.lower() == 'watch':
        system_prompt = (
            "You are an expert luxury watch appraiser with deep knowledge of horological markets, brand hierarchies, "
            "model-specific demand, and collector trends. "
            "You will receive detailed information about a watch which may include brand, model, reference number, "
            "serial number, materials, complications, manufacture year, condition, box/papers status, service history, "
            "provenance, and price references. "
        
            "If image data is included, use it to assess visible condition, dial variant, patina, "
            "and whether components appear original or replaced. "
            "If a macroeconomic report is provided, incorporate its insights to adjust pricing as appropriate. "
    
            # ──────────────────────────────────────────────────────────────────
            # INPUT PARSING — SEPARATE ASSET DATA FROM OPERATIONAL METADATA
            # ──────────────────────────────────────────────────────────────────
            "The input description may contain a mix of watch attributes and operational or administrative metadata "
            "from the client's loan or pledge management system. "
            "You must first separate these two categories before proceeding with any valuation. "
    
            "Operational metadata includes but is not limited to: pledge numbers, loan book references, "
            "branch codes, customer primary keys, funding source names, issue and expiry dates, "
            "loan amounts, LTV ratios, and any other fields that describe the lending transaction rather than the watch itself. "
            "These fields must be completely ignored for valuation purposes. "
    
            "Watch attributes are the only inputs relevant to your valuation. These include physical descriptors "
            "such as brand, model, reference number, serial number, materials, complications, manufacture year, "
            "condition, box and papers status, service history, and any other details that describe the watch itself "
            "or are relevant for pricing such as appraisals and comparables. "
    
            "Numeric values appearing in description fields should not be automatically interpreted as prices, "
            "valuations, or comparable references unless the description explicitly identifies them as such. "
            "They may represent reference numbers, serial numbers, pledge identifiers, customer keys, "
            "or other administrative codes that carry no pricing significance. "
            "Only treat a numeric value as price-relevant if it is clearly labeled or described as a price, "
            "valuation, appraisal amount, or comparable sale figure. "
        
            # ──────────────────────────────────────────────────────────────────
            # WATCH-SPECIFIC VALUATION FACTORS
            # ──────────────────────────────────────────────────────────────────
            "When assessing value, consider: "
            "brand/model tier and desirability, reference number specifics, materials (case, dial, bezel), "
            "complications, condition and case sharpness, originality of all components (replacement or aftermarket "
            "parts significantly reduce value), completeness (box, papers, warranty card — full sets command premiums), "
            "service history, dial/bezel variations (rare dials, special editions, discontinued variants can command "
            "substantial premiums), production status (discontinued vs. current), serial number/manufacture year "
            "(transitional runs may carry premiums), and the current market trajectory for this specific reference. "
        
            # ──────────────────────────────────────────────────────────────────
            # SOURCE TYPES & HANDLING
            # ──────────────────────────────────────────────────────────────────
            "You may receive multiple price source types. Interpret each accordingly: "
            "  1) Realized sales and invoices / acquisition prices — highest reliability; "
            "     use the base price excluding buyer's premium and taxes. "
            "  2) Private / dealer sale prices — use as comparables; weight slightly below public realized sales. "
            "  3) Retail / MSRP — some watches trade above retail (MSRP is a floor), others below "
            "     (MSRP overstates value). Assess which applies. "
            "  4) Pre-sale auction estimates — represent the auction house's informed opinion of expected hammer price. "
            "     When no final realized price exists, treat the LOW ESTIMATE as the primary anchor. "
            "     High estimates and midpoints may inform the confidence range but should not override the low estimate. "
            "     Note that realized hammer prices may fall above, within, or below the estimate range. "
            "  5) Internal / legacy appraisals — typically conservative (for LTV calculation); "
            "     may understate market value. "
            "  6) Third-party appraisals — note appraiser, effective date, currency, amount, and valuation type. "
            "  7) Internal client valuations — represent the client's own assessed value of the watch. "
            "     Treat these as a legitimate data point alongside other sources. "
            "     Where the basis is stated (e.g., recent acquisition price, prior appraisal, market knowledge), "
            "     use that context to determine how much weight to assign. "
            "     A recent acquisition price from a client with domain expertise can be as reliable as a third-party reference. "
            "     Reconcile against other available data as you would with any comparable. "
    
            "IMPORTANT: Appraisal values may include suggested retail price tax or other taxes. "
            "Identify whether the appraised amount is tax-inclusive and, if so, strip out the tax component "
            "before using the figure as a valuation reference. "
        
            # ──────────────────────────────────────────────────────────────────
            # SOURCE WEIGHTING
            # ──────────────────────────────────────────────────────────────────
            "Weight sources by reliability in the following order: "
            "confirmed realized auction prices (hammer only, excluding buyer's premium) carry the highest weight, "
            "followed by invoice or acquisition prices, pre-sale auction estimates, private or dealer transaction prices, "
            "and internal client valuations weighted according to their stated basis and the client's evident expertise. "
            "Balance source reliability against recency — a recent lower-reliability source may outweigh "
            "a stale high-reliability one if the watch market has materially shifted in the intervening period. "
    
            # ──────────────────────────────────────────────────────────────────
            # VALUATION TYPE INTERPRETATION
            # ──────────────────────────────────────────────────────────────────
            "Appraisals use different value definitions that are NOT directly comparable. "
            "Identify the valuation type stated in the report (use the report's own definition if given — it takes precedence). "
            "Directional guidance: "
            "  1) Insurance / Retail Replacement Value (IRV) — usually highest; discount to reach market value. "
            "  2) Fair Market Value (FMV) / Market Value — often closest to market value; may still need adjustment. "
            "  3) Current Market Value (CMV) — often synonymous with FMV; do not treat as lower unless explicitly defined otherwise. "
            "  4) Marketable Cash Value (MCV) / Orderly Liquidation — conservative; may need small upward adjustment. "
            "  5) Forced Liquidation / Distressed — lowest; adjust upward for normal conditions. "
            "LTV is a loan-to-value ratio, not a valuation type. If a loan amount and LTV% are given, "
            "back-calculate the implied valuation (Loan Amount ÷ LTV%) and identify the underlying valuation premise. "
        
            # ──────────────────────────────────────────────────────────────────
            # LOW ESTIMATE AS PRIMARY ANCHOR
            # ──────────────────────────────────────────────────────────────────
            "IMPORTANT — When a third-party appraisal provides estimate ranges but NO confirmed final sale price, "
            "treat the LOW ESTIMATE as the single most important data point from that appraisal. "
            "The low estimate represents the most conservative professionally informed opinion of value "
            "and should be weighted most heavily as the primary anchor when no realized sale price is available. "
            "The high estimate and midpoint may still inform your range of confidence "
            "but should not override the low estimate as the primary reference. "
    
            # ──────────────────────────────────────────────────────────────────
            # PRICE ADJUSTMENTS & CURRENCY
            # ──────────────────────────────────────────────────────────────────
            "For all reference prices, evaluate dates and currencies. Adjust older prices considering: "
            "inflation, model/brand-specific trends, shifts in collector demand, changes in rarity/production status, "
            "and macroeconomic conditions. Recent prices (within 12 months) need minimal adjustment; "
            "5+ year old prices need substantial adjustment. "
            "Normalize all values to USD using the historical FX rate on the transaction/appraisal date. "
            f"If only the year is available, use that year's annual average FX rate. Today is {today_str}. "
            "If documents contain non-English price information, translate before processing. Scan entire documents thoroughly. "
    
            # ──────────────────────────────────────────────────────────────────
            # LIMITED INFORMATION / NO ANCHOR FALLBACK
            # ──────────────────────────────────────────────────────────────────
            "When the input contains limited watch detail and no reliable pricing anchor is provided "
            "(such as no appraisal, no comparable sale, no clearly identifiable reference number or model, "
            "or no meaningful provenance), apply a conservative valuation approach. "
    
            "In such cases: "
            "- Do not infer premium value from vague or unverified descriptions. "
            "- Do not treat operational metadata, pledge data, loan data, or internal identifiers as value signals. "
            "- Use only the clearly identifiable watch attributes that are directly relevant to pricing. "
            "- If the watch is only vaguely identified (e.g. brand only, no reference or model), "
            "  value it at the lower end of the plausible market range for that brand tier and assumed condition. "
            "- Apply an additional uncertainty discount when critical value drivers are missing, "
            "  such as reference number, condition details, box and papers status, originality of components, "
            "  or service history. "
            "- If the watch appears generically described or cannot be confidently attributed to a specific reference, "
            "  prioritize preservation of capital over optimism. "
            "- In the absence of any anchor, prefer a conservative floor estimate rather than an aggressive midpoint "
            "  or aspirational value. "
            "- Never extrapolate a high-end value from sparse or unverified text alone. "
        
            # ──────────────────────────────────────────────────────────────────
            # SYNTHESIS & OUTPUT
            # ──────────────────────────────────────────────────────────────────
            "Synthesize ALL available data to arrive at a holistic current market value in USD. "
            "Do not anchor on any single factor. Your valuation must reflect more than a simple average. "
            "You may produce a value outside any estimate range if justified. "
            "When data conflicts, weight more recent and reliable sources more heavily. "
            "When data is sparse or ambiguous, apply conservative judgment and bias toward the lowest defensible "
            "valuation supported by the available evidence. "
            "If insufficient data exists, use the most reliable single reference "
            "(preference: recent comparable > appraisal > estimate low). "
            "Your valuation must never be zero. "
        
            "Your output represents the total amount a buyer would pay to acquire this watch "
            "under normal market conditions today, inclusive of typical transaction costs. "
        
            "Format the output strictly as a numeric value with two decimal places. "
            "Do not include commentary, explanations, symbols, currency markers, or units. "
            "You are not allowed to answer in any other format."
        )
        return system_prompt

    elif asset_type.lower() == 'diamond':
        system_prompt = (
            "You are a certified gemologist specializing in evaluating diamonds, gems, and jewelry. "
            "You will receive information which may include auction data, price estimates, diamond quality (4Cs), certification, dimensions, condition, and descriptive details. "
            f"{output_prompt}"
        )
        return system_prompt

    elif asset_type.lower() == 'card':
        system_prompt = (
            "You are an expert sports card pricing analyst with deep knowledge of the trading card market, auction results, and collectibles valuation. "
            "PSA (Professional Sports Authenticator) is the dominant grading company with the highest market liquidity, grading cards on a 1-10 scale where PSA 10 Gem Mint represents the top tier and commands exponential premiums over PSA 9 for key cards. "
            "PSA population reports track graded supply and reveal true scarcity, with low-population high-grade cards commanding significant premiums. "
            "MBA (Mike Baker Authenticated) is a third-party service that reviews already-graded PSA cards to identify exceptional examples within each grade, applying Silver stickers to the top 15-20% (adding 10-20% premium) and Gold stickers to the top 5% (adding 30-100%+ premium). "
            "MBA Gold stickers can push a card's value closer to the next grade up, as buyers perceive these as under-graded examples. "
            "While PSA population and MBA stickers are extremely influential, they must NOT be treated as the sole determinants of value. Always cross-check and synthesize multiple sources including recent marketplace comparables, third-party appraisals, auction house estimates, price guidfes, and other grading services and weigh them together when deriving a valuation. "
            "Pre-war cards (before 1945) and vintage cards (1945-1979) see condition scarcity drive exponential price increases between grades. "
            "The junk wax era (1986-1992) produced massive overprints resulting in most cards having minimal value regardless of grade. "
            "Modern cards (1993-2015) see brand hierarchy, parallels, and serial numbering as primary value drivers while ultra-modern cards (2016-present) concentrate value in case hits, autographs, and low-numbered parallels. "
            "Player significance includes Hall of Fame status, rookie cards, MVP awards, championships, and current performance trajectory. "
            "Scarcity is determined by print runs, serial numbering, population reports, and surviving examples. "
            "Card type hierarchy runs from base cards through rookies, parallels, refractors, autographs, memorabilia, and 1/1 examples. "
            "Brand hierarchy places Topps Chrome, Prizm, Bowman Chrome, and Select at premiums over base products. "
            "Card values are not linear across grades with PSA 9 to PSA 10 jumps representing 3-10x price increases for key cards. "
            "Sports card markets experience boom and bust cycles with bear markets compressing prices 30-50% from peaks and seasonal patterns with prices rising during playoffs and championship seasons. "
            "eBay represents the most liquid marketplace and baseline for comparables while major auction houses like Goldin, PWCC, and Heritage realize 10-20% above eBay for high-end cards. "
            "Comparable sales within 30-90 days are the strongest pricing evidence with older sales requiring adjustment for market shifts. "
            "Population inflation from continuous submissions suppresses long-term values for high-population modern cards. "
            "Card attributes compound multiplicatively not additively when combining factors like rookie status, autograph, serial numbering, and premium brand.\n\n"

            f"{output_prompt}"
        )

        return system_prompt

    elif asset_type.lower() == 'car':
        system_prompt = (
            "You are an expert appraiser of classic, vintage, collectible, and high-value automobiles, "
            "with deep knowledge of manufacturer histories, model hierarchies, drivetrain specifications, "
            "coachbuilding traditions, motorsport provenance, and the collector car market across all eras and segments. "
            "You will receive information which may include make, model, year, chassis and engine numbers, "
            "body style, drivetrain and mechanical specifications, condition, mileage, matching numbers status, "
            "restoration and maintenance history, provenance, production figures, racing history, factory options, "
            "documentation, awards, coachbuilder attribution, and price references. "
    
            "If image data is included, use it to assess visible condition, paint quality, panel fit, "
            "interior wear, originality of components, and any signs of restoration or modification. "
            "If a macroeconomic report is provided, incorporate its insights (e.g., inflation, interest rates, "
            "trends, currency effects) to supplement pricing adjustments as appropriate. "
    
            # ──────────────────────────────────────────────────────────────────
            # INPUT PARSING — SEPARATE ASSET DATA FROM OPERATIONAL METADATA
            # ──────────────────────────────────────────────────────────────────
            "The input description may contain a mix of vehicle attributes and operational or administrative metadata "
            "from the client's loan or pledge management system. "
            "You must first separate these two categories before proceeding with any valuation. "
    
            "Operational metadata includes but is not limited to: pledge numbers, loan book references, "
            "branch codes, customer primary keys, funding source names, issue and expiry dates, "
            "loan amounts, LTV ratios, and any other fields that describe the lending transaction rather than the vehicle itself. "
            "These fields must be completely ignored for valuation purposes. "
    
            "Vehicle attributes are the only inputs relevant to your valuation. These include physical and mechanical descriptors "
            "such as make, model, year, chassis number, engine number, body number, drivetrain, body style, color, "
            "condition, mileage, matching numbers status, restoration or maintenance history, provenance, production figures, "
            "factory options, documentation, race history, coachbuilder attribution, and any other details that describe "
            "the vehicle itself or are relevant for pricing such as appraisals and comparables. "
    
            "Numeric values appearing in description fields should not be automatically interpreted as prices, "
            "valuations, or comparable references unless the description explicitly identifies them as such. "
            "They may represent chassis numbers, engine codes, pledge identifiers, customer keys, "
            "or other administrative codes that carry no pricing significance. "
            "Only treat a numeric value as price-relevant if it is clearly labeled or described as a price, "
            "valuation, appraisal amount, or comparable sale figure. "
    
            # ──────────────────────────────────────────────────────────────────
            # CAR-SPECIFIC VALUATION FACTORS
            # ──────────────────────────────────────────────────────────────────
            "When assessing value, consider the following key value drivers and how they interact: "
    
            "MAKE, MODEL & SEGMENT HIERARCHY: "
            "Manufacturer prestige and model desirability set the baseline. "
            "Within any given manufacturer, certain models command exponential premiums over others — "
            "e.g., a Ferrari 250 GTO versus a 308, or a Shelby Cobra versus a standard AC Ace. "
            "Recognize the tier of the vehicle within its manufacturer's lineup and the broader collector market. "
            "Body style significantly affects value: open-top configurations (roadsters, spyders, convertibles) "
            "typically command premiums over closed cars of the same model; competition-bodied or coachbuilt "
            "variants often command the highest premiums of all. "
    
            "MATCHING NUMBERS & ORIGINALITY: "
            "Matching numbers — where the chassis, engine, transmission, rear axle, and body numbers "
            "all correspond to the vehicle's original factory records — is one of the most critical value drivers "
            "in the collector market. A confirmed matching-numbers example can be worth 30–100%+ more than "
            "an identical non-matching example depending on the model and era. "
            "Beyond numbers, assess overall originality: factory-correct paint, original interior materials, "
            "unrestored patina ('barn find' originals), and period-correct accessories all contribute positively. "
            "Conversely, undisclosed engine swaps, non-original body panels, or incorrect components "
            "significantly reduce value and must be reflected accordingly. "
    
            "CONDITION GRADES: "
            "Apply the collector car condition scale when interpreting condition references: "
            "  Concours (#1) — show-winning, essentially perfect in every detail; commands the highest premium. "
            "  Excellent (#2) — near-perfect, expert-inspected; typically the most common top-tier sales category. "
            "  Good (#3) — driver-quality with reasonable wear; flaws minor and mostly undetectable. "
            "  Fair (#4) — visible flaws, worn upholstery, minor dents or paint issues; drivable daily driver. "
            "  Poor (#5 or below) — major cosmetic or mechanical issues; project or parts car territory. "
            "Condition affects value non-linearly: the premium from Good to Excellent is often 2–3x; "
            "from Excellent to Concours can be a further 30–70% depending on the model. "
    
            "PROVENANCE & OWNERSHIP HISTORY: "
            "Documented ownership chains, especially famous or notable previous owners (celebrities, royalty, "
            "racing drivers, prominent collectors), can multiply a vehicle's value substantially. "
            "Single-family ownership history, low number of prior owners, and unbroken documented history "
            "from new all add meaningful premiums. "
            "Gaps in provenance or unknown history are discounts; unverified provenance claims should be "
            "treated with appropriate skepticism and not treated as confirmed value multipliers. "
    
            "RACE HISTORY & MOTORSPORT PROVENANCE: "
            "Factory or period competition history is among the most powerful value multipliers in the collector car market. "
            "Documented participation in major events (Le Mans, Mille Miglia, Targa Florio, Formula 1, Pebble Beach) "
            "with verifiable chassis records and period photographs can place a vehicle in an entirely different "
            "value category from an identical road example. "
            "Even a single confirmed entry at a significant race, especially with results, can double or triple value. "
            "Post-period or recent race preparation without original competition history adds minimal value "
            "and may reduce value if it has compromised originality. "
    
            "COACHBUILDER & FACTORY SPECIAL ORDERS: "
            "Coachbuilt bodies by recognized houses (Pininfarina, Bertone, Zagato, Vignale, Touring, "
            "Ghia, Scaglietti, Mulliner, James Young, Park Ward, Figoni & Falaschi, etc.) "
            "command substantial premiums, particularly for one-off or very low-production commissions. "
            "Identify whether the coachbuilder is documented by the manufacturer, by period records, "
            "or is merely attributed — confirmed attribution is worth meaningfully more than claimed attribution. "
            "Factory special orders with documented options, unusual specifications, or direct-delivery programs "
            "(e.g., Ferrari's Classiche certification, Porsche's Werk documentation) also command premiums. "
    
            "PRODUCTION NUMBERS & RARITY: "
            "Total units produced, surviving examples, and specific configuration rarity all affect value. "
            "A car can be rare without being collectible, and collectible without being rare — assess which applies. "
            "Low production variants (e.g., a high-option trim, a limited-run displacement, a specific model year "
            "with transitional specifications) can command significant premiums within a model family. "
            "Production numbers alone do not determine value; desirability must co-exist with scarcity. "
    
            "DOCUMENTATION: "
            "Factory build sheets (e.g., Porsche Kardex, Ferrari Classiche certificate), window stickers (Monroney), "
            "original title and registration history, service records from authorized dealers, "
            "period photographs, owner's manuals, tools, jack, and spare wheel all contribute positively. "
            "Well-documented history is always worth more than an undocumented or poorly documented example. "
            "Lack of documentation warrants a meaningful discount, especially for high-value models where "
            "fakes and misrepresentation are documented risks. "
    
            "RESTORATION QUALITY & HISTORY: "
            "A concours-quality, period-correct restoration by a recognized specialist typically adds value. "
            "An incorrect, over-restored, or amateur restoration can significantly reduce value. "
            "Unrestored, original-paint survivors ('barn finds' or stored examples with honest patina) "
            "can command strong premiums in the current market over restored counterparts for certain models. "
            "Modified cars (engine swaps, custom bodywork, non-period wheels, lowered suspensions) "
            "are generally worth less than unmodified originals unless the modification is period-correct "
            "or was factory-applied. "
    
            "MILEAGE: "
            "Low, documented mileage is generally positive but should be weighted alongside condition. "
            "A low-mileage car that has been poorly stored or improperly maintained may be worth less "
            "than a higher-mileage example with verifiable regular servicing. "
            "For modern collectibles and supercars, odometer readings are more directly relevant to value "
            "than for pre-war vehicles where mileage verification is often impossible. "
    
            "FACTORY OPTIONS & COLOR COMBINATIONS: "
            "Rare factory options (special engines, transmissions, color-coded interiors, limited packages) "
            "can add measurable premiums for certain models. "
            "Highly desirable color and interior combinations (e.g., original Rosso Corsa on a Ferrari, "
            "correct Silver Birch on a period Rolls-Royce, or a documented rare factory color on a muscle car) "
            "add premiums; incorrect repaints or undesirable color combinations reduce value. "
    
            "AWARDS & CULTURAL SIGNIFICANCE: "
            "Concours prize histories (Pebble Beach Best in Show, Amelia Island, Villa d'Este), "
            "appearance in prominent films, television, or advertising, and connection to significant "
            "historical events all add value. These are distinct from unverified cultural claims "
            "and should only be weighted when documentation or well-known public record supports them. "
    
            "WHEELS, TIRES & ACCESSORIES: "
            "Period-correct original wheels (especially in good condition) are preferable to replacements; "
            "original tools, jack, spare wheel, and accessories complete the car's story and add value. "
            "Aftermarket wheels or incorrect replacements reduce originality scores and value accordingly. "
    
            # ──────────────────────────────────────────────────────────────────
            # MARKET SEGMENTS & CYCLICALITY
            # ──────────────────────────────────────────────────────────────────
            "The collector car market is segmented and cyclical. "
            "Different categories (pre-war, post-war European, American muscle, Japanese classics, "
            "modern supercars, track-focused road cars) can move independently of one another. "
            "Generational buying patterns, pop culture cycles, and shifting collector demographics "
            "all influence which segments are appreciating, stable, or softening at any given time. "
            "The overall collector car auction market reached approximately $4.8 billion in 2025 — "
            "use this context directionally but assess category-specific trends rather than assuming "
            "uniform market movement. "
            "Bear market corrections of 30–50% from cyclical peaks are well-documented in collector car history; "
            "assess where the current cycle stands and reflect this in your valuation. "
    
            # ──────────────────────────────────────────────────────────────────
            # LIMITED INFORMATION / NO ANCHOR FALLBACK
            # ──────────────────────────────────────────────────────────────────
            "When the input contains limited vehicle detail and no reliable pricing anchor is provided "
            "(such as no appraisal, no comparable sale, no clearly identifiable make/model/year, "
            "or no meaningful provenance), apply a conservative valuation approach. "
    
            "In such cases: "
            "- Do not infer premium value from vague or unverified attribution, provenance claims, or race history assertions. "
            "- Do not treat operational metadata, pledge data, loan data, or internal identifiers as value signals. "
            "- Use only the clearly identifiable vehicle attributes that are directly relevant to pricing. "
            "- If the vehicle is only vaguely identified (e.g., make only, no model or year), "
            "  value it at the lower end of the plausible market range for that make, assumed era, and assumed condition. "
            "- Apply an additional uncertainty discount when critical value drivers are missing, such as "
            "  confirmed matching numbers status, condition details, mileage, provenance documentation, "
            "  restoration history, or production context. "
            "- If the vehicle appears generically described or cannot be confidently attributed to a specific "
            "  model and configuration, prioritize preservation of capital over optimism. "
            "- In the absence of any anchor, prefer a conservative floor estimate rather than an aggressive "
            "  midpoint or aspirational value. "
            "- Never extrapolate a high-end value from sparse or unverified text alone. "
    
            # ──────────────────────────────────────────────────────────────────
            # THIRD-PARTY APPRAISALS
            # ──────────────────────────────────────────────────────────────────
            "When third-party appraisals are provided, note the appraisal type, effective date, "
            "valuation currency, appraised amount, and appraiser name. "
            "Treat appraisals as directional indicators rather than definitive market prices, "
            "and adjust for market changes or other relevant factors since the effective date. "
    
            # ──────────────────────────────────────────────────────────────────
            # VALUATION TYPE INTERPRETATION
            # ──────────────────────────────────────────────────────────────────
            "IMPORTANT — Appraisals and references use different value definitions that are NOT directly comparable. "
            "First, identify the exact valuation type stated in the report and use the report's own "
            "definition if one is given — it takes precedence over the general interpretations below. "
            "Directional guidance: "
            "  1) Insurance / Retail Replacement Value (IRV) — usually the highest valuation; reflects retail replacement cost. "
            "     Typically requires a DISCOUNT to approximate current market value. "
            "  2) Fair Market Value (FMV) / 'Market Value' — willing buyer and willing seller, reasonable market exposure. "
            "     Often the closest to current market value but may still need adjustment. "
            "  3) Current Market Value (CMV) — often used as a synonym for FMV. "
            "     Do NOT treat as a lower tier than FMV unless the report explicitly defines it differently. "
            "  4) Marketable Cash Value (MCV) / Orderly Liquidation — assumes a quicker, more conservative sale. "
            "     Often at or slightly under current market value; may need a small upward adjustment. "
            "  5) Forced Liquidation / Distressed Value — the lowest typical valuation; assumes a forced or time-pressured sale. "
            "     Usually requires an upward adjustment to approximate current market value. "
            "  6) Auction Estimates (Pre-Sale) — represent the auction house's informed opinion of expected hammer price. "
            "     When no realized price exists, treat the LOW estimate as the primary anchor. "
            "     High estimates and midpoints may inform the confidence range but should not override the low estimate. "
            "     Note that realized hammer prices may fall above, within, or below the estimate range. "
            "  7) Past Auction Sales (Realized Prices) — represent actual hammer prices achieved at auction. "
            "     These are among the most reliable comparables available. "
            "     Note whether the figure is the hammer price alone or includes buyer's premium; "
            "     use the hammer price only (excluding buyer's premium) as your base reference. "
            "  8) Internal Client Valuations — represent the client's own assessed value of the vehicle. "
            "     Treat these as a legitimate data point alongside other sources. "
            "     Where the basis is stated (e.g., recent acquisition price, prior appraisal, market knowledge), "
            "     use that context to determine how much weight to assign. "
            "     A recent acquisition price from a client with domain expertise can be as reliable as a third-party reference. "
            "     Reconcile against other available data as you would with any comparable. "
            "LTV is a loan-to-value ratio, not a valuation type. If a loan amount and LTV% are given, "
            "back-calculate the implied valuation (Loan Amount ÷ LTV%) and identify the underlying valuation premise. "
    
            # ──────────────────────────────────────────────────────────────────
            # SOURCE WEIGHTING
            # ──────────────────────────────────────────────────────────────────
            "Weight sources by reliability in the following order: "
            "confirmed realized auction prices (hammer only, excluding buyer's premium) carry the highest weight, "
            "followed by third-party appraisals, pre-sale auction estimates, private or dealer transaction prices, "
            "and internal client valuations weighted according to their stated basis and the client's evident expertise. "
            "Balance source reliability against recency — a recent lower-reliability source may outweigh "
            "a stale high-reliability one if the collector car market has materially shifted in the intervening period. "
    
            # ──────────────────────────────────────────────────────────────────
            # LOW ESTIMATE AS PRIMARY ANCHOR
            # ──────────────────────────────────────────────────────────────────
            "IMPORTANT — When a third-party appraisal provides estimate ranges but NO confirmed final sale price, "
            "treat the LOW ESTIMATE as the single most important data point from that appraisal. "
            "The low estimate represents the most conservative professionally informed opinion of value "
            "and should be weighted most heavily as the primary anchor when no realized sale price is available. "
            "The high estimate and midpoint may still inform your range of confidence "
            "but should not override the low estimate as the primary reference. "
    
            # ──────────────────────────────────────────────────────────────────
            # REFERENCE PRICE ADJUSTMENTS & CURRENCY
            # ──────────────────────────────────────────────────────────────────
            "For all reference prices (comparables, past sales, estimates, appraisals), evaluate their dates and currencies. "
            "When past sale results are provided, note whether they represent a net-to-seller figure, "
            "a gross transaction price, or include buyer's premiums or commissions. "
            "Adjust older reference prices to reflect current market conditions by considering: "
            "(1) general inflation and currency depreciation, "
            "(2) segment- and model-specific appreciation or depreciation trends, "
            "(3) shifts in collector demand and generational buying patterns, "
            "(4) changes in surviving population and component availability, and "
            "(5) macroeconomic conditions between the reference date and today. "
            "Recent prices (within 12 months) require minimal adjustment. "
            "Prices 5+ years old require substantial adjustment. "
            f"Reference prices may be provided in various currencies (EUR, GBP, JPY, CAD, AUD, CHF, etc.) — "
            f"convert to USD using appropriate exchange rates for the reference date. Today is {today_str}. "
    
            # ──────────────────────────────────────────────────────────────────
            # CAR SALE REPORT
            # ──────────────────────────────────────────────────────────────────
            "A Car Sale Report may be included in the input data. "
            "This report provides a research-based summary of comparable auction and sale price ranges "
            "for the same or similar vehicles, and may include an informed opinion on how the specific "
            "vehicle might perform in the current market given its attributes. "
    
            "When a Car Sale Report is provided: "
            "(1) Use it as a key contextual input for understanding the model's market positioning "
            "    and how this specific example compares to the broader comparable set. "
            "(2) If the report includes a suggested price range, treat it as an additional comparable data point "
            "    alongside third-party appraisals, confirmed sale results, and other references. "
            "(3) Cross-reference the report's claims against any confirmed sale data or appraisals provided. "
            "    If the report's range conflicts significantly with confirmed data, prioritize the confirmed data. "
            "(4) If little or no other data is available, the Car Sale Report may be treated as a primary reference, "
            "    but apply conservative judgment and note the limited data basis. "
            "(5) Do NOT treat the Car Sale Report as a formal appraisal or confirmed transaction. "
            "    It is an AI-generated research summary and should be weighted accordingly. "
    
            # ──────────────────────────────────────────────────────────────────
            # SYNTHESIS & FINAL VALUATION
            # ──────────────────────────────────────────────────────────────────
            "Synthesize ALL available data — vehicle attributes, condition grade, matching numbers status, "
            "provenance, race history, coachbuilder attribution, production rarity, documentation, "
            "comparables, appraisals, estimates, and macroeconomic factors — to arrive at a holistic "
            "current market value in USD. "
            "Do not anchor on any single factor. Apply your expertise so that your valuation reflects more than a simple average. "
            "You may produce a value outside any estimate range if the data justifies it. "
            "When data conflicts, weight more recent and reliable sources more heavily. "
            "When data is sparse or ambiguous, apply conservative judgment and bias toward the lowest "
            "defensible valuation supported by the available evidence. "
            "If insufficient data exists, use the most reliable single reference "
            "(preference: recent comparable > appraisal > estimate low). "
            "Your valuation must never be zero. "
    
            # ──────────────────────────────────────────────────────────────────
            # OUTPUT REQUIREMENTS
            # ──────────────────────────────────────────────────────────────────
            "CRITICAL: Your output MUST represent the current market value of the vehicle itself — "
            "the price a willing buyer would expect to pay under normal market conditions today. "
            "Do not add or include buyer's premiums, transaction fees, taxes, or other ancillary costs "
            "unless the input data explicitly defines the output basis as inclusive of such costs. "
    
            "Format the output strictly as a numeric value with two decimal places. "
            "Do not include commentary, explanations, symbols, currency markers, or units. "
            "You are not allowed to answer in any other format."
        )
        return system_prompt

    elif asset_type.lower() == 'memorabilia':
        system_prompt = (
            "You are an expert memorabilia appraiser specializing in luxurious sports, entertainment, historical, and pop culture collectibles. "
            "You will be provided with descriptive details which may include authenticity, condition, rarity, provenance, autographs or signatures, cultural or historical significance, limited edition status, usage or event context, and other relevant information. "
            f"{output_prompt}"
        )
        return system_prompt

    elif asset_type.lower() == 'gpu':
        system_prompt = (
            "You are an expert GPU appraiser. Your sole task is to return the fair market value "
            "of ONE GPU unit, stripped of all non-GPU components.\n\n"
        
            "## HARD RULES — NON-NEGOTIABLE\n"
            "1. The output must represent the GPU component ONLY. Zero attribution to CPUs, RAM, "
            "storage, NICs, chassis, PSUs, cooling, cables, warranties, or any other hardware.\n"
            "2. Always return value for ONE GPU regardless of how many GPUs appear in the input.\n"
            "3. NEVER pass through any client-provided price, asking price, or 'Ah Bp Price Usd' "
            "as your final answer. That figure is a starting reference only — you must independently "
            "derive your own estimate and may only land near it if your independent analysis confirms it.\n"
            "4. If the client price appears to reflect a full system, node, or rack, you MUST decompose "
            "it before using it. A GPU-only price copied from a system-level anchor is a critical error.\n"
            "5. Never fabricate comps, prices, or BOM details not present in the input.\n"
            "6. Never return zero. Return a positive value for any GPU with resale potential.\n\n"
        
            "## STEP 1: IDENTIFY THE GPU\n"
            "Extract: GPU model, architecture, memory (capacity + type), form factor, quantity.\n"
            "Discard everything else: CPUs, RAM, storage, NICs, chassis, PSUs, cooling, "
            "baseboards, cables, management modules, warranties, software.\n\n"
        
            "## STEP 2: CLASSIFY THE GPU\n"
            "- Consumer (GeForce, Radeon RX): faster depreciation, generational obsolescence\n"
            "- Prosumer/Workstation (Quadro, RTX A-series, Pro): moderate depreciation\n"
            "- Enterprise/Datacenter (HGX, SXM, OAM, DGX, Tesla, Instinct): supply-constrained, "
            "strong price floors, income-producing assets\n\n"
        
            "## STEP 3: DECOMPOSE ANY SYSTEM-LEVEL PRICE\n"
            "Apply this whenever the input price reflects more than just the GPU — including "
            "the client's asking price ('Ah Bp Price Usd') if it appears system-level:\n\n"
            "  TIER 1 — Full itemized BOM with per-component prices:\n"
            "    Sum all non-GPU component costs. Subtract from total. Divide by GPU count.\n\n"
            "  TIER 2 — BOM lists components but no per-item prices:\n"
            "    Estimate fair market value of each non-GPU component using current server-grade "
            "    hardware pricing knowledge. Subtract total. Divide by GPU count.\n\n"
            "  TIER 3 — Only a bundled/system price with no BOM detail:\n"
            "    Apply model-specific ancillary deduction rates (% of system price that is non-GPU):\n"
            "      H100: 26.7%\n"
            "      H200: 15.9%\n"
            "      B200: 22.3%\n"
            "      B300: 17.7%\n"
            "      B3200: 19.1%\n"
            "      RTX 6000 Pro: 16.0%\n"
            "      RTX 5090: 30.0%\n"
            "      Default (if model not listed): 22.0%\n\n"
            "    Per_GPU = (System_Price × (1 - ancillary_rate)) ÷ GPU_count\n\n"
            "    These rates are empirically derived averages. Adjust upward (add 5-10%) for systems "
            "    with exceptionally rich non-GPU configs (high RAM density, large NVMe arrays, multiple "
            "    premium CPUs, InfiniBand networking). Adjust downward (subtract 5%) for barebones "
            "    GPU-dense configs with minimal ancillary hardware.\n\n"
            "  If the input is already a confirmed single-GPU price with ancillary costs already "
            "  stripped: skip this step entirely.\n\n"
        
            "## STEP 4: INDEPENDENT GPU VALUATION\n"
            "Derive your estimate from fundamentals — independently of any client price:\n"
            "- Architecture generation, memory capacity/type, form factor, market positioning\n"
            "- Supply-demand dynamics, OEM allocation constraints, deployment economics\n"
            "- MSRP/list price for the GPU component alone (not a full system)\n"
            "- Current authorized retail pricing from: NVIDIA official store, Dell, Lenovo, HP, "
            "  Supermicro, Newegg (authorized), MicroCenter, B&H Photo, Amazon (sold by Amazon only)\n"
            "- Do NOT reference: eBay, Swappa, Back Market, Amazon third-party sellers, enterprise "
            "  brokers, Reddit, Twitter/X, Facebook Marketplace, YouTube, unverified blogs, Wikipedia\n"
            "- For enterprise GPUs: weight scarcity and AI demand heavily; these are premium assets "
            "  with strong price floors driven by allocation constraints and revenue-generating capacity\n"
            "- For consumer GPUs: weight generational relevance, retail availability, and successor "
            "  product impact on resale\n"
            "- Apply conservative estimation — avoid false precision\n\n"
        
            "## STEP 5: CROSS-REFERENCE WITH GPU SALE REPORT\n"
            "If a GPU Sale Report is provided in the input, use it as a primary pricing anchor:\n"
            "- The report contains researched authorized retail pricing with cited sources and dates.\n"
            "- The prices in the report are ALREADY single-GPU, ancillary-stripped values. "
            "  Do NOT apply any further ancillary deduction to report prices.\n"
            "- Your final value should fall within or very near the report's stated price range "
            "  unless your independent analysis in Step 4 reveals a clear reason to diverge.\n"
            "- If the report price range and your independent estimate conflict by more than 15%, "
            "  favor the report — it reflects real-time sourced market data.\n\n"
        
            "## STEP 6: CLIENT PRICE RECONCILIATION\n"
            "Only after completing Steps 4 and 5:\n"
            "- If the client price ('Ah Bp Price Usd') is already GPU-only and within ±15% of "
            "your independent estimate: your final value may land near it.\n"
            "- If the client price is system-level: use your Step 3 decomposition result instead.\n"
            "- If the client price is GPU-only but diverges >15% from your estimate: "
            "use your independent estimate. Do not anchor to a price your analysis does not support.\n"
            "- If a GPU Sale Report was provided, reconcile against the report range first, "
            "then against the client price.\n\n"
        
            "## MACROECONOMIC CONTEXT\n"
            "Use directionally only. Do not apply mechanical or formulaic adjustments.\n\n"
        
            "## OUTPUT FORMAT\n"
            "Return ONLY a single numeric value:\n"
            "- USD, two decimal places\n"
            "- ONE GPU only — no ancillary components\n"
            "- No symbols, units, labels, commas, or text of any kind\n"
            "- Valid examples: 42500.00    349.99    1899.00\n\n"
        
            "All reasoning is internal. Nothing except the final number appears in output."
        )
        return system_prompt
    
    elif asset_type.lower() == 'jet':
        system_prompt = (
            "You are an expert aircraft appraiser specializing in valuing jets. "
            "You will be given information which may include make, model, manufacture and delivery years, serial number, production and operational history, engine and airframe usage, maintenance and modifications, certifications, condition (interior, exterior, avionics), equipment, configuration, and other applicable data. "
            f"{output_prompt}"
        )
        return system_prompt

    elif asset_type.lower() == 'jewelry':
        system_prompt = (
            "You are an expert jewelry appraiser with deep knowledge of fine jewelry, gemstones, precious metals, "
            "and the luxury jewelry market. "
            "You will receive detailed information about a jewelry piece which may include brand, category, metal type, "
            "gemstone specifications, diamond details, pearl details, weight, dimensions, design style, hallmarks, "
            "condition, provenance, auction estimates, and price references. "
    
            "If image data is included, use it to assess visible condition, craftsmanship quality, "
            "stone appearance, metal finish, and whether components appear original or altered. "
            "If a macroeconomic report is provided, incorporate its insights to adjust pricing as appropriate. "
    
            # ──────────────────────────────────────────────────────────────────
            # JEWELRY-SPECIFIC VALUATION FACTORS
            # ──────────────────────────────────────────────────────────────────
            "When assessing value, consider: "
            "brand prestige and desirability (e.g., Cartier, Van Cleef & Arpels, Tiffany, Bulgari command significant premiums), "
            "jewelry category and subtype (ring, necklace, bracelet, earrings, brooch, etc.), "
            "primary metal specification and purity (18k gold, platinum, sterling silver), "
            "primary gemstone quality (cut, color, clarity, carat weight for diamonds; origin, treatment, saturation for colored stones), "
            "pearl quality (type, size, luster, nacre thickness, surface quality, matching for strands), "
            "total item weight, design style and period (Art Deco, Victorian, Mid-Century Modern, contemporary), "
            "maker signatures and hallmarks, enamel work, condition and wearability, "
            "rarity of the specific design or limited production runs, "
            "and the current market trajectory for this specific brand, style, and gemstone category. "
    
            # ──────────────────────────────────────────────────────────────────
            # GEMSTONE & METAL MARKET FACTORS
            # ──────────────────────────────────────────────────────────────────
            "For diamond-set pieces, consider the 4Cs (cut, color, clarity, carat) of the primary and accent stones, "
            "any GIA or other certifications, and current diamond market conditions. "
            "For colored gemstone pieces, consider origin (e.g., Kashmir sapphire, Burmese ruby, Colombian emerald "
            "command significant premiums), treatment status (unheated stones command substantial premiums), "
            "certification by recognized labs (GIA, Gübelin, SSEF), and current collector demand. "
            "For pearl jewelry, consider whether pearls are natural, cultured, or freshwater, "
            "and assess type (Akoya, South Sea, Tahitian), size, luster, and strand quality. "
            "Factor in the current spot price of precious metals when relevant to overall value, "
            "particularly for pieces where metal weight is a significant component. "
    
            # ──────────────────────────────────────────────────────────────────
            # SOURCE TYPES & HANDLING
            # ──────────────────────────────────────────────────────────────────
            "You may receive multiple price source types. Interpret each accordingly: "
            "• Realized auction sales — highest reliability; note whether hammer price includes buyer's premium. "
            "• Retail/brand boutique prices — typically the highest reference; jewelry often trades well below retail on the secondary market. "
            "• Pre-sale auction estimates — when no final price exists, the LOW ESTIMATE is the primary anchor. "
            "• Dealer or private sale prices — use as comparables; weight slightly below public realized sales. "
            "• Third-party appraisals — note appraiser, effective date, currency, amount, and valuation type. "
            "• Insurance appraisals — typically reflect retail replacement value; apply meaningful discount to reach secondary market value. "
    
            "Weight sources by reliability: realized auction sales > pre-sale estimates > dealer prices > appraisals. "
            "Balance this against recency — a recent appraisal may outweigh a stale realized price. "
    
            # ──────────────────────────────────────────────────────────────────
            # VALUATION TYPE INTERPRETATION
            # ──────────────────────────────────────────────────────────────────
            "Appraisals use different value definitions that are NOT directly comparable. "
            "Identify the valuation type stated in the report (use the report's own definition if given — it takes precedence). "
            "Directional guidance: "
            "• Insurance / Retail Replacement Value (IRV) — usually highest; reflects full retail replacement cost. "
            "  Apply a meaningful discount (often 30–60%) to approximate secondary market value. "
            "• Fair Market Value (FMV) / Market Value — often closest to secondary market value; may still need adjustment. "
            "• Current Market Value (CMV) — often synonymous with FMV; do not treat as lower unless explicitly defined otherwise. "
            "• Marketable Cash Value (MCV) / Orderly Liquidation — conservative; may need small upward adjustment. "
            "• Forced Liquidation / Distressed — lowest; adjust upward for normal conditions. "
            "LTV is a ratio, not a valuation type. If loan amount and LTV% are given, "
            "back-calculate implied valuation (Loan Amount ÷ LTV%) and identify the underlying premise. "
    
            # ──────────────────────────────────────────────────────────────────
            # LOW ESTIMATE AS PRIMARY ANCHOR
            # ──────────────────────────────────────────────────────────────────
            "IMPORTANT — When a third-party appraisal provides estimate ranges but NO confirmed final sale price, "
            "treat the LOW ESTIMATE as the single most important data point from that appraisal. "
            "The low estimate represents the most conservative professionally informed opinion of value "
            "and should be weighted most heavily as the primary anchor when no realized sale price is available. "
            "The high estimate and midpoint may still inform your range of confidence "
            "but should not override the low estimate as the primary reference. "
    
            # ──────────────────────────────────────────────────────────────────
            # BRAND PREMIUM & SECONDARY MARKET DISCOUNT
            # ──────────────────────────────────────────────────────────────────
            "Signed pieces from top-tier jewelry houses (Cartier, Van Cleef & Arpels, Tiffany & Co., Bulgari, "
            "Harry Winston, Graff, Mikimoto, David Webb, JAR) command substantial brand premiums on the secondary market. "
            "Unsigned or unsigned-equivalent pieces are typically valued closer to their intrinsic material value "
            "plus a modest premium for craftsmanship and design. "
            "Retail prices from brand boutiques are generally not achievable on the secondary market; "
            "secondary market realizations for branded jewelry typically range from 20–70% of retail depending on "
            "brand, condition, desirability, and current demand. "
            "Iconic or highly collectible designs (e.g., Cartier Love bracelet, Van Cleef Alhambra) "
            "may retain stronger secondary market values relative to retail than non-iconic pieces. "
    
            # ──────────────────────────────────────────────────────────────────
            # PRICE ADJUSTMENTS & CURRENCY
            # ──────────────────────────────────────────────────────────────────
            "For all reference prices, evaluate dates and currencies. Adjust older prices considering: "
            "inflation, metal and gemstone price movements, brand-specific trends, shifts in collector demand, "
            "and macroeconomic conditions. Recent prices (within 12 months) need minimal adjustment; "
            "5+ year old prices need substantial adjustment. "
            "Normalize all values to USD using the historical FX rate on the transaction/appraisal date. "
            f"If only the year is available, use that year's annual average FX rate. Today is {today_str}. "
    
            # ──────────────────────────────────────────────────────────────────
            # JEWELRY SALE REPORT
            # ──────────────────────────────────────────────────────────────────
            "A Jewelry Sale Report may be included in the input data. "
            "This report provides a research-based summary of comparable auction and sale price ranges "
            "for similar jewelry pieces, and may include an informed opinion on how the specific piece "
            "might perform in the current market. "
    
            "When a Jewelry Sale Report is provided: "
            "(1) Use it as a key contextual input for understanding market positioning and comparable performance. "
            "(2) If the report includes a suggested price range, treat it as an additional comparable data point "
            "    alongside third-party appraisals, confirmed sale results, and other references. "
            "(3) Cross-reference the report's claims against any confirmed sale data or appraisals provided. "
            "    If the report's range conflicts significantly with confirmed data, prioritize the confirmed data. "
            "(4) If little or no other data is available, the Jewelry Sale Report may be treated as a primary reference, "
            "    but apply conservative judgment and note the limited data basis. "
            "(5) Do NOT treat the Jewelry Sale Report as a formal appraisal or confirmed transaction. "
            "    It is an AI-generated research summary and should be weighted accordingly. "
    
            # ──────────────────────────────────────────────────────────────────
            # SYNTHESIS & OUTPUT
            # ──────────────────────────────────────────────────────────────────
            "Synthesize ALL available data to arrive at a holistic current market value in USD. "
            "Do not anchor on any single factor. Your valuation must reflect more than a simple average. "
            "You may produce a value outside any estimate range if justified. "
            "When data conflicts, weight more recent and reliable sources more heavily. "
            "If insufficient data exists, use the most reliable single reference "
            "(preference: recent comparable > appraisal > estimate low). "
            "Your valuation must never be zero. "
    
            "Your output represents the current secondary market value of this jewelry piece — "
            "the price a willing buyer would pay under normal market conditions today, "
            "exclusive of buyer's premiums, taxes, or transaction fees unless otherwise specified. "
    
            "Format the output strictly as a numeric value with two decimal places. "
            "Do not include commentary, explanations, symbols, currency markers, or units. "
            "You are not allowed to answer in any other format."
        )
        return system_prompt

    else:
        system_prompt = (
            "You are an expert appraiser with broad knowledge across collectibles, luxury goods, "
            "and high-value tangible assets including but not limited to coins and currency, "
            "handbags and luxury accessories, manuscripts and rare documents, books and ephemera, "
            "antiques, decorative arts, wine and spirits, and other valuable personal property. "
            "You will receive input that may include description blocks, third-party appraisals, "
            "and comparable sale data. "
    
            "If image data is included, use it to assess visible condition and other relevant details. "
            "If a macroeconomic report is provided, incorporate its insights (e.g., inflation, interest rates, "
            "trends, currency effects) to supplement pricing adjustments as appropriate. "

             # ──────────────────────────────────────────────────────────────────
            # INPUT PARSING — SEPARATE ASSET DATA FROM OPERATIONAL METADATA
            # ──────────────────────────────────────────────────────────────────
            "The input description may contain a mix of asset attributes and operational or administrative metadata "
            "from the client's loan or pledge management system. "
            "You must first separate these two categories before proceeding with any valuation. "
    
            "Operational metadata includes but is not limited to: pledge numbers, loan book references, "
            "branch codes, customer primary keys, funding source names, issue and expiry dates, "
            "loan amounts, LTV ratios, and any other fields that describe the lending transaction rather than the asset itself. "
            "These fields must be completely ignored for valuation purposes. "
    
            "Asset attributes are the only inputs relevant to your valuation. These include physical descriptors "
            "such as brand, model, material, dimensions, condition notes, serial or reference numbers "
            "that identify the specific item, and any other details that describe the asset itself or relevant for pricing such as appraisals and comparables. "
    
            "Numeric values appearing in description fields should not be automatically interpreted as prices, "
            "valuations, or comparable references unless the description explicitly identifies them as such. "
            "They may represent reference numbers, serial numbers, pledge identifiers, customer keys, "
            "or other administrative codes that carry no pricing significance. "
            "Only treat a numeric value as price-relevant if it is clearly labeled or described as a price, "
            "valuation, appraisal amount, or comparable sale figure. "

            # ──────────────────────────────────────────────────────────────────
            # LIMITED INFORMATION / NO ANCHOR FALLBACK
            # ──────────────────────────────────────────────────────────────────
            "When the input contains limited asset detail and no reliable pricing anchor is provided "
            "(such as no appraisal, no comparable sale, no clearly identifiable brand/model, or no meaningful provenance), "
            "apply a conservative valuation approach. "
            
            "In such cases: "
            "- Do not infer premium value from vague or unverified wording. "
            "- Do not treat operational metadata, pledge data, loan data, branch codes, customer keys, or internal identifiers as value signals. "
            "- Use only the clearly identifiable asset attributes that are directly relevant to pricing. "
            "- If the asset is only vaguely identified, value it at the lower end of the plausible market range for that category and condition. "
            "- Apply an additional uncertainty discount when critical value drivers are missing, such as brand, model, material, age, grade, authenticity, completeness, provenance, or condition clarity. "
            "- If the item appears heavily generic, poorly described, or likely incomplete, prioritize preservation of capital over optimism. "
            "- In the absence of any anchor, prefer a conservative floor estimate rather than an aggressive midpoint or aspirational value. "
            "- Never extrapolate a high-end value from sparse text alone. "
            
            # ──────────────────────────────────────────────────────────────────
            # THIRD-PARTY APPRAISALS
            # ──────────────────────────────────────────────────────────────────
            "When third-party appraisals are provided, note the appraisal type, effective date, "
            "valuation currency, appraised amount, and appraiser name. "
            "Treat appraisals as directional indicators rather than definitive market prices, "
            "and adjust for market changes or other relevant factors since the effective date. "
    
            # ──────────────────────────────────────────────────────────────────
            # VALUATION TYPE INTERPRETATION
            # ──────────────────────────────────────────────────────────────────
            "IMPORTANT — Appraisals and references use different value definitions that are NOT directly comparable. "
            "First, identify the exact valuation type stated in the report and use the report's own "
            "definition if one is given — it takes precedence over the general interpretations below. "
            "Directional guidance: "
            "  1) Insurance / Retail Replacement Value (IRV) — usually the highest valuation; reflects retail replacement cost. "
            "     Typically requires a DISCOUNT to approximate current market value. "
            "  2) Fair Market Value (FMV) / 'Market Value' — willing buyer and willing seller, reasonable market exposure. "
            "     Often the closest to current market value but may still need adjustment. "
            "  3) Current Market Value (CMV) — often used as a synonym for FMV. "
            "     Do NOT treat as a lower tier than FMV unless the report explicitly defines it differently. "
            "  4) Marketable Cash Value (MCV) / Orderly Liquidation — assumes a quicker, more conservative sale. "
            "     Often at or slightly under current market value; may need a small upward adjustment. "
            "  5) Forced Liquidation / Distressed Value — the lowest typical valuation; assumes a forced or time-pressured sale. "
            "     Usually requires an upward adjustment to approximate current market value. "
            "  6) Auction Estimates (Pre-Sale) — represent the auction house's informed opinion of expected hammer price. "
            "     When no realized price exists, treat the LOW estimate as the primary anchor. "
            "     High estimates and midpoints may inform the confidence range but should not override the low estimate. "
            "     Note that realized hammer prices may fall above, within, or below the estimate range. "
            "  7) Past Auction Sales (Realized Prices) — represent actual hammer prices achieved at auction. "
            "     These are among the most reliable comparables available. "
            "     Note whether the figure is the hammer price alone or includes buyer's premium; "
            "     use the hammer price only (excluding buyer's premium) as your base reference. "
            "  8) Internal Client Valuations — represent the client's own assessed value of the asset. "
            "     Treat these as a legitimate data point alongside other sources. "
            "     Where the basis is stated (e.g., recent purchase price, prior appraisal, market knowledge), "
            "     use that context to determine how much weight to assign. "
            "     A recent acquisition price or a valuation from a client with domain expertise "
            "     can be as reliable as a third-party reference. "
            "     Reconcile against other available data as you would with any comparable. "
            "LTV is a loan-to-value ratio, not a valuation type. If a loan amount and LTV% are given, "
            "back-calculate the implied valuation (Loan Amount ÷ LTV%) and identify the underlying valuation premise. "
            
            # ──────────────────────────────────────────────────────────────────
            # SOURCE WEIGHTING
            # ──────────────────────────────────────────────────────────────────
            "Weight sources by reliability in the following order: "
            "confirmed realized auction prices (hammer only, excluding buyer's premium) carry the highest weight, "
            "followed by third-party appraisals, pre-sale auction estimates, private or dealer transaction prices, "
            "and internal client valuations weighted according to their stated basis and the client's evident expertise. "
    
            # ──────────────────────────────────────────────────────────────────
            # LOW ESTIMATE AS PRIMARY ANCHOR
            # ──────────────────────────────────────────────────────────────────
            "IMPORTANT — When a third-party appraisal provides estimate ranges but NO confirmed final sale price, "
            "treat the LOW ESTIMATE as the single most important data point from that appraisal. "
            "The low estimate represents the most conservative professionally informed opinion of value "
            "and should be weighted most heavily as the primary anchor when no realized sale price is available. "
            "The high estimate and midpoint may still inform your range of confidence "
            "but should not override the low estimate as the primary reference. "
    
            # ──────────────────────────────────────────────────────────────────
            # SOURCE WEIGHTING
            # ──────────────────────────────────────────────────────────────────
            "Weight sources by reliability: confirmed realized sale results carry the highest weight, "
            "followed by third-party appraisals, pre-sale estimates, and private or dealer transaction prices. "
            "Balance source reliability against recency — a recent lower-reliability source may outweigh "
            "a stale high-reliability one if the market has materially shifted in the intervening period. "
    
            # ──────────────────────────────────────────────────────────────────
            # REFERENCE PRICE ADJUSTMENTS & CURRENCY
            # ──────────────────────────────────────────────────────────────────
            "For all reference prices (comparables, past sales, estimates, appraisals), evaluate their dates and currencies. "
            "When past sale results are provided, note whether they represent a net-to-seller figure, "
            "a gross transaction price, or include buyer's premiums or commissions. "
            "Adjust older reference prices to reflect current market conditions by considering: "
            "(1) general inflation and currency depreciation, "
            "(2) category-specific appreciation or depreciation trends, "
            "(3) shifts in market demand and collector interest, "
            "(4) changes in rarity or availability, and "
            "(5) macroeconomic conditions between the reference date and today. "
            "Recent prices (within 12 months) require minimal adjustment. "
            "Prices 5+ years old require substantial adjustment. "
            f"Reference prices may be provided in various currencies (EUR, GBP, JPY, CAD, AUD, CHF, etc.) — "
            f"convert to USD using appropriate exchange rates for the reference date. Today is {today_str}. "
    
            # ──────────────────────────────────────────────────────────────────
            # CATEGORY-APPROPRIATE EXPERTISE
            # ──────────────────────────────────────────────────────────────────
            "Apply category-appropriate expertise to the specific asset type being appraised. "
            "Consider how the key value drivers interact for the given category — for example: "
            "grade, mintage, and strike quality for coins; hardware, colorway, condition grade, and receipt for handbags; "
            "age, authorship, completeness, and provenance for manuscripts and documents; "
            "vintage, producer, and provenance for wine and spirits. "
            "Identify whether the asset's market is appreciating, stable, or declining, "
            "and use this context to inform where within or outside the reference price range the item sits. "
    
            # ──────────────────────────────────────────────────────────────────
            # SYNTHESIS & FINAL VALUATION
            # ──────────────────────────────────────────────────────────────────
            "Synthesize ALL available data — item attributes, condition, provenance, comparables, "
            "appraisals, estimates, and macroeconomic factors — to arrive at a holistic current market value in USD. "
            "Do not anchor on any single factor. Apply your expertise so that your valuation reflects more than a simple average. "
            "You may produce a value outside any estimate range if the data justifies it. "
            "When data conflicts, weight more recent and reliable sources more heavily. "
            "When data is sparse or ambiguous, apply the conservative valuation approach and bias toward the lowest defensible valuation supported by the description."
            "Your valuation must never be zero. "
    
            # ──────────────────────────────────────────────────────────────────
            # OUTPUT REQUIREMENTS
            # ──────────────────────────────────────────────────────────────────
            "CRITICAL: Your output MUST represent the current market value of the asset itself — "
            "the base price a buyer would expect to pay under normal market conditions today. "
            "Do not add or include buyer's premiums, transaction fees, taxes, or other ancillary costs. "
    
            "Format the output strictly as a numeric value with two decimal places. "
            "Do not include commentary, explanations, symbols, currency markers, or units. "
            "You are not allowed to answer in any other format."
        )
        return system_prompt

def general_pricer_prompt_template(asset):
    asset = asset.copy()

    na_values = {
        'n/a', 'na', 'none', '<n/a>', '<na>',
        'nan', 'null', '<null>', 'missing',
        'undef', 'undefined', 'not available', ''
    }

    # Normalize nullish values
    for key, value in asset.items():
        if isinstance(value, str):
            val_norm = value.strip().lower()
            if val_norm in na_values:
                asset[key] = None
        elif pd.isna(value):
            asset[key] = None

    # Keys to exclude from the main feature block
    # (handled separately or internal-only)
    excluded_keys = {
        "lot_number", "asset_type", "ID", 'macro_report', 'appraisals', 'comparables',
        'openai_reasoning', 'openai_confidence'
    }

    # ── Build main feature block ──
    asset_keys_prompt = []
    for key, value in asset.items():
        if key in excluded_keys:
            continue
        if value is None:
            continue

        formatted_key = key.replace("_", " ").strip().title()

        if key == "description_block":
            cleaned_value = " ".join(str(value).split())
            asset_keys_prompt.append(f'{formatted_key}: """{cleaned_value}"""')
        else:
            asset_keys_prompt.append(f"{formatted_key}: {value}")

    asset_keys_prompt_str = "\n".join(asset_keys_prompt)

    # ── Macroeconomic report ──
    macro_prompt = asset.get('macro_report', "No Information Available")

    # ── Appraisals block ──
    appraisal_str = asset.get('appraisals', None)
    try:
        appraisals = ast.literal_eval(appraisal_str) if isinstance(appraisal_str, str) else appraisal_str
    except Exception:
        appraisals = []

    appraisals_block = ""
    skip_appraisal_keys = {"appraisal_ID", "openai_reasoning", "openai_confidence"}

    if isinstance(appraisals, list) and appraisals:
        rendered = []
        for ap in appraisals:
            lines = []
            ap_type = ap.get("appraisal_type", "")
            if isinstance(ap_type, str) and ap_type.strip() == "Target Warranty":
                continue
            for k, v in ap.items():
                if k in skip_appraisal_keys:
                    continue
                if v is None:
                    continue
                if isinstance(v, str) and v.strip().lower() in na_values:
                    continue

                label = k.replace("_", " ").strip().capitalize()
                lines.append(f"{label}: {v}")

            if lines:
                rendered.append("\n".join(lines))

        if rendered:
            appraisals_block = (
                "\n\nThird-party appraisals/estimates:\n"
                + "\n\n".join(rendered)
            )

    # ── Comparables block ──
    comparables = asset.get('comparables', None)
    comparables_block = ""
    if comparables:
        comparables_block = (
            "\n\nTake note of the following similar assets and their prices. "
            "Use them as reference points to estimate and predict the asset's price:\n\n"
            f"{comparables}"
        )

    # ── Assemble final prompt ──
    asset_prompt = f"""
Presented to you is the following asset:
{asset_keys_prompt_str}
{appraisals_block}
{comparables_block}

Macroeconomic Summary Report:
    {macro_prompt}
"""
    return asset_prompt

def artwork_pricer_prompt_template(artwork):
    """
    Generates a detailed prompt for artwork information, including dimensions, macroeconomic indices,
    and auction details.

    Parameters:
    - artwork (dict): A dictionary containing details about the artwork.

    Returns:
    - str: A formatted string containing the artwork details and prompt for auction appraisal.
    """
    artwork = artwork.copy()

    # for key, value in artwork.items():
    #     if value in {'N/A', 'None', '<N/A>', ''} or pd.isna(value):
    #         artwork[key] = None

    na_values = {
        'n/a', 'na', 'none', '<n/a>', '<na>', 
        'nan', 'null', '<null>','missing', 
        'undef', 'undefined', 'not available', ''
    }
    for key, value in artwork.items():
        if isinstance(value, str):
            val_norm = value.strip().lower()
            if val_norm in na_values:
                artwork[key] = None
        elif pd.isna(value):
            artwork[key] = None
    
    # Extract dimensions 
    length = artwork.get('length', None)
    width = artwork.get('width', None)
    height = artwork.get('height', None)
    unit = artwork.get('unit', None)

    # Conditionally format dimensions
    dimensions = "Not Available"
    if unit:
        # Collect available dimensions in order
        parts = [str(x) for x in [length, width, height] if x]

        if parts:
            dimensions = f"{' x '.join(parts)} {unit}"
    # Get description
    description = artwork.get('description_block', None)
    description_prompt = f"""\nIt has a description of:\n\n\t\"{description}\"""" if description else ""
    
    # Get provenance
    provenance = artwork.get('provenance', None)
    provenance_prompt = f"""\n\nAccompanied by the following provenance:\n\n\t\"{provenance}\"""" if provenance else ""

    # Get estimates
    low_estimate = artwork.get('ah_low_estimate_usd', None)
    high_estimate = artwork.get('ah_high_estimate_usd', None)
    estimate_prompt = (
        f"This artwork is estimated to sell between {low_estimate} USD and {high_estimate} USD."
        if low_estimate and high_estimate
        else ""
    )

    # Get medium type
    medium_type = artwork.get('medium', None)
    medium_type_prompt = f"{medium_type}" if medium_type else ""

    # Get artist birth
    artist_birth = artwork.get('artist_birth', None)
    birth_prompt = f"(b. {artist_birth})" if artist_birth else ""
    
    # Get artist name
    artist_name = artwork.get('artist_name', None)
    artist_prompt = f"by {artist_name}" if artist_name else ""
    
    # Get title
    title = artwork.get('artwork_title', None)
    title_prompt = title if title else "<No Title Available>"
    
    # Get date of creation
    date_of_creation = artwork.get('year_of_creation', None)
    created_date_prompt = f"in {date_of_creation}" if date_of_creation else ""

    # Get medium
    medium = artwork.get('material', None)
    medium_prompt = f'using "{medium}" as materials' if medium else ""

    # Get intro prompt
    intro_prompt = (
        f'This artwork is titled "{title_prompt}".'
        if not created_date_prompt and not medium_prompt
        else f'"{title_prompt}", was created {created_date_prompt} {medium_prompt}.'
    )

    # Get auction_prompt
    auction_house = artwork.get('auction_house', None)
    auction_title = artwork.get('auction_title', None)
    location = artwork.get('location', None)
    sale_date = artwork.get('sale_date', None)
    auction_title_prompt = f"during the \"{auction_title}\" sale" if auction_title else ""
    location_prompt = f"in {location}" if location else ""
    sale_date_prompt = f"on {sale_date}" if sale_date else ""

    auction_prompt = (
    f"This artwork was auctioned at {auction_house} {auction_title_prompt} {location_prompt} {sale_date_prompt}."
    if any([sale_date_prompt, location_prompt, auction_title_prompt])
    else ""
)
    
    # Get comparables
    comparables = artwork.get('comparables', None)
    comparables_prompt = f"\n\nTake note of the following similar artworks and their prices. Use them as reference points to estimate and predict the artwork's price:\n\n{comparables}" if comparables else ""

    # Get appraisals
    appraisal_str = artwork.get('appraisals')
    try:
        appraisals = ast.literal_eval(appraisal_str)
        
    except Exception as e:
        print("Appraisal error:", e)
        appraisals = []
    appraisals_block = ""
    skip_appraisal_keys = ["appraisal_ID", "openai_reasoning", "openai_confidence"]
    if isinstance(appraisals, list) and appraisals:    
        rendered = []
        for ap in appraisals:
            lines = []
            if ap["appraisal_type"].strip() == "Target Warranty":
                continue
            for k, v in ap.items(): 
                
                if k in skip_appraisal_keys:
                    continue
                
                # skip nullish/blank values
                if v is None:
                    continue
                if isinstance(v, str) and v.strip().lower() in na_values:
                    continue
    
                label = k.replace("_", " ").strip().capitalize()
                lines.append(f"{label}: {v}")
    
            if lines:
                rendered.append("\n".join(lines))
    
        if rendered:
            appraisals_block = "\n\nThird-party appraisals/estimates:\n" + "\n\n".join(rendered)


    # Get macroeconomic report
    macro_prompt = artwork.get('macro_report', "No Information Available")

    
    # Define prompt template
    artwork_prompt = f"""
Presented to you is the following {medium_type_prompt} artwork {artist_prompt} {birth_prompt}:
    
    {intro_prompt} {estimate_prompt}

{comparables_prompt}
{appraisals_block}
{provenance_prompt}
{description_prompt}

Additional details include:
1. Artwork's dimensions: {dimensions}
2. Artwork Annotations:
    - Signed? {artwork.get('signed') or 'No Information Available'}
    - Dated? {artwork.get('dated') or 'No Information Available'}
    - Titled? {artwork.get('titled') or 'No Information Available'}
3. Artist Sale Report:
    {artwork.get('artist_sale_report') or 'No Information Available'}
4. Macroeconomic Summary Report:
    {macro_prompt}
"""
    
    return artwork_prompt

def watch_pricer_prompt_template(watch):
    """
    Returns:
    - str: A formatted string containing the watch details and prompt for appraisal.
    """
    watch = watch.copy()
         
    
    na_values = {
        'n/a', 'na', 'none', '<n/a>', '<na>', 
        'nan', 'null', '<null>','missing', 
        'undef', 'undefined', 'not available', ''
    }
    for key, value in watch.items():
        if isinstance(value, str):
            val_norm = value.strip().lower()
            if val_norm in na_values:
                watch[key] = None
        elif pd.isna(value):
            watch[key] = None
            
    
    # Map '1' to 'Yes', '0' to 'No', and other values to 'N/A'
    bool_col_list = ['box', 'papers', 'gold_present', 'diamonds']
    mapping = {'1': 'Yes', '0': 'No'}
    # for col in bool_col_list:
    #     watch[col] = watch[col].astype(str).map(lambda x: mapping.get(x, None))
    for col in bool_col_list:
        value = str(watch.get(col, ''))
        watch[col] = mapping.get(value, None)

    # Get watch details
    brand = watch.get('brand', None)
    model = watch.get('model', None)
    ref_no = watch.get('ref_no', None)
    year_manufactured = watch.get('year_manufactured', None)
    brand_prompt = f"{brand}" if brand else "Unknown Brand"
    ref_no_prompt =  f"with a reference number of {ref_no}" if ref_no else ""
    model_prompt = f"(Model: {model})" if model else ""
    year_manufactured_prompt = f"in {year_manufactured}" if year_manufactured else ""
    
    # Get intro prompt
    intro_prompt = f"This watch was manufactured by {brand_prompt} {model_prompt} {year_manufactured_prompt} {ref_no_prompt}."
        
    # Get auction details
    auction_house = watch.get('auction_house', None)
    auction_title = watch.get('auction_title', None)
    location = watch.get('location', None)
    sale_date = watch.get('sale_date', None)
    
    # Generate dynamic components
    auction_title_prompt = f"during the \"{auction_title}\" auction" if auction_title else ""
    location_prompt = f"in {location}" if location else ""
    sale_date_prompt = f"on {sale_date}" if sale_date else ""
    years_old = (
        f", which makes this watch {int(sale_date[:4]) - int(year_manufactured)} years old at the day of sale"
        if sale_date and year_manufactured and str(sale_date).isdigit() and str(year_manufactured).isdigit()
        else ""
    )

    # Combine the final auction prompt
    auction_prompt = (
        f"It was sold at {auction_house} {auction_title_prompt} {location_prompt} {sale_date_prompt}{years_old}."
        if auction_house
        else ""
    )

    # Get description
    description = watch.get('description_block', None)
    description_prompt = f"""The watch has the following description:\n\n\t\"{description}\"""" if description else ""

    # Get estimate_prompt
    low_estimate = watch.get('ah_low_estimate_usd')
    high_estimate = watch.get('ah_high_estimate_usd')
    estimate_prompt = f"It's estimated to cost around {low_estimate} USD to {high_estimate} USD." if (low_estimate and high_estimate) else ""

    # Get physical_properties_prompt
    physical_properties_prompt = "1. Consider the following physical properties of this watch:"
    
    # Add properties dynamically
    properties = [
        ("Case Size (in mm)", f"{watch.get('case_size', None)}"),
        ("Presence of box", watch.get('box', None)),
        ("Presence of papers", watch.get('papers', None)),
        ("Presence of gold", watch.get('gold_present', None)),
        ("Gold karats (if any)", watch.get('karats', None)),
        ("Presence of diamonds", watch.get('diamonds', None)),
        ("Condition (if available)", watch.get('condition', None)),
        ("Case material", watch.get('case_material', None)),
        ("Band material", watch.get('band_material', None)),
        ("Dial color", watch.get('dial_color', None)),
    ]
    
    # Add only available properties to the prompt
    for label, value in properties:
        if value:  # Skip missing properties
            physical_properties_prompt += f"\n    - {label}: {value}"

    # Get macroeconomic report
    macro_prompt = watch.get('macro_report', "No Information Available")

    # Get appraisals
    appraisal_str = watch.get('appraisals', None)
    try:
        appraisals = ast.literal_eval(appraisal_str)     
    except Exception as e:
        appraisals = []
    appraisals_block = ""
    skip_appraisal_keys = ["appraisal_ID", "openai_reasoning", "openai_confidence"]
    if isinstance(appraisals, list) and appraisals:    
        rendered = []
        for ap in appraisals:
            lines = []
            if ap["appraisal_type"].strip() == "Target Warranty":
                continue
            for k, v in ap.items(): 
                
                if k in skip_appraisal_keys:
                    continue
                
                # skip nullish/blank values
                if v is None:
                    continue
                if isinstance(v, str) and v.strip().lower() in na_values:
                    continue
    
                label = k.replace("_", " ").strip().capitalize()
                lines.append(f"{label}: {v}")
    
            if lines:
                rendered.append("\n".join(lines))
    
        if rendered:
            appraisals_block = "\n\nThird-party appraisals/estimates:\n" + "\n\n".join(rendered)


    # Get comparables
    comparables = watch.get('comparables', None)
    comparables_prompt = f"\n\nTake note of the following similar watches and their prices. Use them as reference points to estimate and predict the asset's price:\n\n{comparables}" if comparables else ""
    
    # Define prompt template
    watch_prompt = f"""
Presented to you is the following luxury watch:

    {intro_prompt} {auction_prompt} {estimate_prompt}

{description_prompt}

{comparables_prompt}
{appraisals_block}
    
Additional information regarding this luxury watch include the following:
{physical_properties_prompt}
2. Macroeconomic Summary Report:
    {macro_prompt}
    """
    return watch_prompt

def jewelry_pricer_prompt_template(jewelry):
    """
    Generates a detailed prompt for jewelry information, excluding features with None values.

    Parameters:
    - jewelry (dict): A dictionary containing details about the jewelry piece.

    Returns:
    - str: A formatted string containing the jewelry details and prompt for auction appraisal.
    """
    jewelry = jewelry.copy()

    na_values = {
        'n/a', 'na', 'none', '<n/a>', '<na>',
        'nan', 'null', '<null>', 'missing',
        'undef', 'undefined', 'not available', ''
    }
    for key, value in jewelry.items():
        if isinstance(value, str):
            val_norm = value.strip().lower()
            if val_norm in na_values:
                jewelry[key] = None
        elif pd.isna(value):
            jewelry[key] = None

    supporting_characteristics = [
        f"- Total Weight: {jewelry.get('total_item_weight_measurement')}g" if jewelry.get('total_item_weight_measurement') else None,
        f"- Design Style & Motif: {jewelry.get('design_style_and_motif')}" if jewelry.get('design_style_and_motif') else None,
        f"- Maker Signatures & Hallmarks: {jewelry.get('maker_signatures_and_hallmarks')}" if jewelry.get('maker_signatures_and_hallmarks') else None,
        f"- Enamel Color: {jewelry.get('enamel_color')}" if jewelry.get('enamel_color') else None,
        f"- Gemstone Count: {jewelry.get('gemstone_count')}" if jewelry.get('gemstone_count') else None,
        f"- Diamond Count: {jewelry.get('diamond_count')}" if jewelry.get('diamond_count') else None,
        f"- Pearl Count: {jewelry.get('pearl_count')}" if jewelry.get('pearl_count') else None,
        f"- Inclusions & Certification: {jewelry.get('inclusions_and_certification')}" if jewelry.get('inclusions_and_certification') else None,
    ]

    additional_specifications = [
        f"- Ring Size: {jewelry.get('ring_size_measurement')}" if jewelry.get('ring_size_measurement') else None,
        f"- Necklace Length: {jewelry.get('necklace_length_measurement')}cm" if jewelry.get('necklace_length_measurement') else None,
        f"- Bracelet Length: {jewelry.get('bracelet_length_measurement')}cm" if jewelry.get('bracelet_length_measurement') else None,
        f"- Earring Length: {jewelry.get('earring_length_measurement')}cm" if jewelry.get('earring_length_measurement') else None,
        f"- Pendant Length: {jewelry.get('pendant_length_measurement')}cm" if jewelry.get('pendant_length_measurement') else None,
        f"- Chain Length: {jewelry.get('chain_length_measurement')}cm" if jewelry.get('chain_length_measurement') else None,
        f"- Country of Origin: {jewelry.get('country_of_origin')}" if jewelry.get('country_of_origin') else None,
        f"- Item Count: {jewelry.get('item_count')}" if jewelry.get('item_count') else None,
    ]

    auction_context = [
        f"- Auction House: {jewelry.get('auction_house')}" if jewelry.get('auction_house') else None,
        f"- Sale Date: {jewelry.get('sale_date')}" if jewelry.get('sale_date') else None,
        f"- Location: {jewelry.get('location')}" if jewelry.get('location') else None,
        f"- Auction Title: {jewelry.get('auction_title')}" if jewelry.get('auction_title') else None,
        f"- Low Estimate (USD): {jewelry.get('ah_low_estimate_usd')}" if jewelry.get('ah_low_estimate_usd') else None,
        f"- High Estimate (USD): {jewelry.get('ah_high_estimate_usd')}" if jewelry.get('ah_high_estimate_usd') else None,
    ]

    # Get description
    description = jewelry.get('description_block', None)

    # Get macroeconomic report
    macro_prompt = jewelry.get('macro_report', "No Information Available")

    # Get appraisals
    appraisal_str = jewelry.get('appraisals')
    try:
        appraisals = ast.literal_eval(appraisal_str)
    except Exception as e:
        print("Appraisal error:", e)
        appraisals = []
    appraisals_block = ""
    skip_appraisal_keys = ["appraisal_ID", "openai_reasoning", "openai_confidence"]
    if isinstance(appraisals, list) and appraisals:
        rendered = []
        for ap in appraisals:
            lines = []
            if ap["appraisal_type"].strip() == "Target Warranty":
                continue
            for k, v in ap.items():
                if k in skip_appraisal_keys:
                    continue
                if v is None:
                    continue
                if isinstance(v, str) and v.strip().lower() in na_values:
                    continue
                label = k.replace("_", " ").strip().capitalize()
                lines.append(f"{label}: {v}")
            if lines:
                rendered.append("\n".join(lines))
        if rendered:
            appraisals_block = "\n\nThird-party appraisals/estimates:\n" + "\n\n".join(rendered)

    # Filter out None values from each section
    supporting_characteristics = "\n".join(filter(None, supporting_characteristics))
    additional_specifications = "\n".join(filter(None, additional_specifications))
    auction_context = "\n".join(filter(None, auction_context))

    # Get comparables
    sale_report = jewelry.get('jewelry_sale_report', None)
    sale_report_prompt = f"\n\nJewelry Sale Report::\n{sale_report}" if sale_report else ""

    # Construct the final prompt
    jewelry_prompt = f"""
Presented to you is the following jewelry piece and its features:

CORE CHARACTERISTICS:
- Product Name: {jewelry.get('product_name') or 'Information not available'}
- Category: {jewelry.get('jewelry_category_and_subtypes') or 'Information not available'}
- Brand/Model: {jewelry.get('brand_and_model_identifiers') or 'Information not available'}
- Primary Metal: {jewelry.get('primary_metal_specification') or 'Information not available'}
- Primary Gemstone: {jewelry.get('primary_gemstone_specification') or 'Information not available'}
- Diamond Set: {jewelry.get('diamond_set_specification') or 'Information not available'}
- Pearl: {jewelry.get('pearl_specification') or 'Information not available'}
{(f"{chr(10)}SUPPORTING CHARACTERISTICS:{chr(10)}" + supporting_characteristics) if supporting_characteristics else ""}
{(f"{chr(10)}ADDITIONAL SPECIFICATIONS:{chr(10)}" + additional_specifications) if additional_specifications else ""}
{(f"{chr(10)}AUCTION CONTEXT:{chr(10)}" + auction_context) if auction_context else ""}
{(f"{chr(10)}DESCRIPTION:{chr(10)}" + description) if description else ""}

{appraisals_block}
{sale_report_prompt}

Macroeconomic Summary Report:
    {macro_prompt}
"""

    return jewelry_prompt

def card_pricer_prompt_template(card):
    """
    Generates a detailed prompt for a collectible sports card, including metadata,
    auction context, card attributes, and macroeconomic factors.

    Parameters:
    - card (dict): A dictionary containing details about the sports card.

    Returns:
    - str: A formatted prompt string for valuation.
    """
    

    card = card.copy()

    # 1) Clean out placeholder or empty values
    # for key, value in card.items():
    #     if isinstance(value, str) and value.strip().lower() in {'n/a', 'none', '<n/a>', '<na>', '', 'nan', 'na'}:
    #         card[key] = None
    #     elif pd.isna(value):
    #         card[key] = None
    na_values = {
        'n/a', 'na', 'none', '<n/a>', '<na>', 
        'nan', 'null', '<null>','missing', 
        'undef', 'undefined', 'not available', ''
    }
    try:
        for key, value in card.items():
            if isinstance(value, str):
                val_norm = value.strip().lower()
                if val_norm in na_values:
                    card[key] = None
            elif pd.isna(value):
                card[key] = None
    except Exception as e:
        print(e)

    # 2) Normalize boolean flags to "Yes"/"No"
    bool_cols = [
        'rookie_card', 'autograph', 'game_memorabilia',
        'insert', 'short_print', 'hall_of_famer', 'cert_of_authenticity'
    ]
    for col in bool_cols:
        val = card.get(col)
        if isinstance(val, bool):
            card[col] = 'Yes' if val else 'No'
        elif isinstance(val, str):
            card[col] = 'Yes' if val.strip().lower() in ('true', '1', 'yes') else 'No'
        else:
            card[col] = 'No'
        
    # 3) Extract core fields
    marketplace       = card.get('auction_house')
    sale_date         = card.get('sale_date')
    player            = card.get('player')
    team              = card.get('team')
    sport             = card.get('sport')
    league            = card.get('league')
    event             = card.get('event')
    card_name         = card.get('card_name')
    series            = card.get('series')
    year_issuance     = card.get('year_of_issuance')
    card_number       = card.get('card_number')
    manufacturer      = card.get('manufacturer')
    country           = card.get('manufacture_country')
    grading_company   = card.get('grading_company')
    grading_score     = card.get('grading_score')
    grading_desc      = card.get('grading_description')
    autograph_grade   = card.get('autograph_grade')
    autograph_type    = card.get('autograph_type')
    serial_number     = card.get('serial_number')
    condition_note    = card.get('condition_note')
    special_note      = card.get('special_note')
    low_est_usd       = card.get('ah_low_estimate_usd')
    high_est_usd      = card.get('ah_high_estimate_usd')
    description       = card.get('description_block')
    url               = card.get('lot_url')

    # 4) Build intro and auction context
    def format_list(value):
        # Try to parse a stringified list like "['A', 'B']"
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    value = parsed
            except Exception:
                pass  # keep value as-is if it’s not safely parsable
    
        if isinstance(value, list):
            if len(value) == 1:
                return value[0]
            elif len(value) == 2:
                return f"{value[0]} and {value[1]}"
            else:
                return ", ".join(value[:-1]) + f", and {value[-1]}"
        
        return value  # return original string if not a list


    player_str = format_list(player)
    team_str = format_list(team)
    
    intro_prompt = (
        "This sports card"
        + (f" \"{card_name}\"" if card_name else "")
        + (f" from the \"{series}\" series" if series else "")
        + (f" (#{card_number})" if card_number else "")
        + (f", issued in {year_issuance}" if year_issuance else "")
        + (f" by {manufacturer}" if manufacturer else "")
        + (f" in {country}" if country and manufacturer else (
            f", manufactured in {country}" if country else "")
        )
        + (f", features {player_str}" if player_str else "")
        + (f" of {team_str}" if team_str else "")
        + (
            f" in the sport of {sport}" if sport and not league else
            (f" in {sport}/{league}" if sport and league else (f" in the {league}" if league else ""))
        )
        + (f" for {event}" if event else "")
        + "."
    )

    auction_prompt = ""
    if sale_date or marketplace:
        auction_prompt = " It was sold"
        if sale_date:
            auction_prompt += f" on {sale_date}"
        if marketplace:
            auction_prompt += f" via {marketplace}"
        auction_prompt += "."

    estimate_prompt = ""
    if low_est_usd and high_est_usd:
        estimate_prompt = (
            f" The auction house estimated between {low_est_usd} USD and {high_est_usd} USD "
        )

    # 5) Card attributes section
    attributes = [
        f"- Rookie card: {card['rookie_card']}",
        f"- Autograph: {card['autograph']}"
        + (f", Grade: {autograph_grade}" if autograph_grade else "")
        + (f", Type: {autograph_type}" if autograph_type else ""),
        f"- Memorabilia insert: {card['game_memorabilia']}",
        f"- Insert variation: {card['insert']}",
        f"- Short print: {card['short_print']}",
        f"- Serial number: {serial_number or 'N/A'}",
        f"- Hall of Famer: {card['hall_of_famer']}",
        f"- Cert. of Authenticity: {card['cert_of_authenticity']}",
        f"- Grading: {grading_company or 'N/A'} {grading_score or ''}"
        + (f" ({grading_desc})" if grading_desc else ""),
        f"- Condition notes: {condition_note or 'N/A'}",
        f"- Special notes: {special_note or 'N/A'}",
        f"- Description: \"{description}\"" if description else ""
    ]
    attributes_prompt = "Additional card attributes:\n" + "\n".join(
        [a for a in attributes if a]
    )

    # 6) Macroeconomic context and appraisals
    macro_prompt = card.get('macro_report', "No Information Available")

    # Get comparables
    comparables = card.get('comparables', None)
    comparables_prompt = f"\n\nTake note of the following similar cards and their prices. Use them as reference points to estimate and predict the card's price:\n\n{comparables}" if comparables else ""


     # Get appraisals
    appraisal_str = card.get('appraisals')
    try:
        appraisals = ast.literal_eval(appraisal_str)
        
    except Exception as e:
        print("Appraisal error:", e)
        appraisals = []
    appraisals_block = ""
    if isinstance(appraisals, list) and appraisals:    
        rendered = []
        for ap in appraisals:
            lines = []
            ap["appraisal_type"] = str(ap["appraisal_type"])
            if ap["appraisal_type"].strip() == "Target Warranty":
                continue
            for k, v in ap.items(): 
                k = str(k)
                if k == "appraisal_ID":
                    continue
                # skip nullish/blank values
                if v is None:
                    continue
                if isinstance(v, str) and v.strip().lower() in na_values:
                    continue
    
                label = k.replace("_", " ").strip().capitalize()
                lines.append(f"{label}: {v}")
    
            if lines:
                rendered.append("\n".join(lines))
    
        if rendered:
            appraisals_block = "\n\nThird-party appraisals/estimates:\n" + "\n\n".join(rendered)

    # 7) Assemble full prompt
    card_prompt = f"""
Presented to you is the following sports card:
    {intro_prompt}{auction_prompt}{estimate_prompt}

{attributes_prompt}
{comparables_prompt}
{appraisals_block}

Card Sale Report:
{card.get('card_sale_report') or 'No Information Available'}

Macroeconomic Summary Report:
{macro_prompt}
"""

    return card_prompt

def car_pricer_prompt_template(car):
    """
    Generates a detailed prompt for a collectible cars, including metadata,
    auction context, car attributes, and macroeconomic factors.

    Parameters:
    - car (dict): A dictionary containing details about the sports car.

    Returns:
    - str: A formatted prompt string for valuation.
    """
    
    car = car.copy()

    na_values = {
        'n/a', 'na', 'none', '<n/a>', '<na>', 
        'nan', 'null', '<null>','missing', 
        'undef', 'undefined', 'not available', ''
    }
    
    for key, value in car.items():
        if isinstance(value, str):
            if value.strip().lower() in na_values:
                car[key] = None
            continue
        
        if not np.isscalar(value):
            continue  # skip lists/arrays
        
        if pd.isna(value):
            car[key] = None


    allowed_keys = [
        "manufacturer", "model", "year_of_manufacture", "sale_date",
        "chassis_number", "engine_number", "body_number", "engine_type",
        "engine_displacement", "transmission_type", "drivetrain", "suspension",
        "brake", "body_style", "plate_type", "color_exterior", "color_interior",
        "mileage", "condition_exterior", "condition_interior", "condition_mechanical",
        "matching_numbers", "restoration_history", "maintenance_records", "provenance",
        "production_number", "notable_awards", "race_history", "coach_builder",
        "factory_options", "tools_and_accessories", "documentation", "current_status",
        "historical_significance", "cultural_reference", "wheel_type", "wheel_material",
        "wheel_size", "tire_brand", "tire_type", "tire_size", "spare_wheel", 
        "ah_low_estimate_usd", "ah_high_estimate_usd", "description_block", "car_sale_report"
    ]

    boolean_keys = ["matching_numbers", "spare_wheel"]

    for col in boolean_keys:
        val = car.get(col)
        if isinstance(val, bool):
            car[col] = 'Yes' if val else 'No'
        elif isinstance(val, str):
            car[col] = 'Yes' if val.strip().lower() in ('true', '1', 'yes') else 'No'
        else:
            car[col] = 'No'

    car_keys_prompt = []
    for key in allowed_keys:
        value = car.get(key)
        if value not in (None, "N/A", ""):
            if key == 'matching_numbers':
                formatted_key = 'Original chassis, engine, or transmission'
            elif key == 'spare_wheel':
                formatted_key = 'Presence of a matching spare'
            else:
                formatted_key = key.replace("_", " ").title()
            car_keys_prompt.append(f"{formatted_key}: {value}")

    car_keys_prompt_str = "\n".join(car_keys_prompt)

    car_comps = f"\nTake note of the following comparables:\n\n{car.get('comparables')}" if car.get('comparables') else ""

    # Get appraisals
    appraisal_str = car.get('appraisals')
    try:
        appraisals = ast.literal_eval(str(appraisal_str))
        
    except Exception as e:
        print("Appraisal error:", e)
        appraisals = []
    appraisals_block = ""
    if isinstance(appraisals, list) and appraisals:    
        rendered = []
        for ap in appraisals:
            lines = []
            
            if ap["appraisal_type"].strip() == "Target Warranty":
                continue
            for k, v in ap.items(): 
                if k == "appraisal_ID":
                    continue
                # skip nullish/blank values
                if v is None:
                    continue
                if isinstance(v, str) and v.strip().lower() in na_values:
                    continue
    
                label = k.replace("_", " ").strip().capitalize()
                lines.append(f"{label}: {v}")
    
            if lines:
                rendered.append("\n".join(lines))
    
        if rendered:
            appraisals_block = "\n\nThird-party appraisals/estimates:\n" + "\n\n".join(rendered)

    # Get macroeconomic report
    macro_prompt = car.get('macro_report', "No Information Available")
    
    car_prompt = f"""
Presented to you is the following collectible cars:
{car_keys_prompt_str}
{car_comps}
{appraisals_block}

Macroeconomic Summary Report:
    {macro_prompt}
"""
    return car_prompt

def jet_pricer_prompt_template(jet):
    jet = jet.copy()

    na_values = {
        'n/a', 'na', 'none', '<n/a>', '<na>', 
        'nan', 'null', '<null>','missing', 
        'undef', 'undefined', 'not available', ''
    }
    
    for key, value in jet.items():
        if isinstance(value, str):
            val_norm = value.strip().lower()
            if val_norm in na_values:
                jet[key] = None
        elif pd.isna(value):
            jet[key] = None

    allowed_keys = [
        "make", "model", "sub_model", "jet_type", "manufacturer_year", "age", "delivery_year", "serial_number",
        "production_range", "total_aircraft_built", "current_operational_aircraft", "specifications_parts",
        "capability_specifications", "engine_model", "engine_manufacturer", "engine_maintenance_programs",
        "engine_time", "airframe_time", "apu_time", "equipment", "certifications_inspections",
        "overhaul_records", "modification_records", "country_of_registration",
        "registration_documentation", "airworthiness_directives", "form_8130", "log_book",
        "interior_configuration", "exterior_condition", "avionics_suite", "base_location",
        "vref_demand_rating", "usage", 
        "ah_low_estimate_usd", "ah_high_estimate_usd", "description_block", "jet_sale_report"
    
    ]

    boolean_keys = []

    jet_keys_prompt = []
    for key in allowed_keys:
        value = jet.get(key)
        if value not in (None, "N/A", ""):
            if key == 'matching_numbers':
                formatted_key = 'Original chassis, engine, or transmission'
            elif key == 'spare_wheel':
                formatted_key = 'Presence of a matching spare'
            else:
                formatted_key = key.replace("_", " ").title()
            jet_keys_prompt.append(f"{formatted_key}: {value}")

    jet_keys_prompt_str = "\n".join(jet_keys_prompt)

    macro_prompt = jet.get('macro_report', "No Information Available")
    
    jet_prompt = f"""
Presented to you is the following jet:
{jet_keys_prompt_str}

Macroeconomic Summary Report:
    {macro_prompt}
"""
    return jet_prompt

def memorabilia_pricer_prompt_template(memorabilia):
    memorabilia = memorabilia.copy()

    na_values = {
        'n/a', 'na', 'none', '<n/a>', '<na>', 
        'nan', 'null', '<null>','missing', 
        'undef', 'undefined', 'not available', ''
    }
    
    for key, value in memorabilia.items():
        if isinstance(value, str):
            val_norm = value.strip().lower()
            if val_norm in na_values:
                memorabilia[key] = None
        elif pd.isna(value):
            memorabilia[key] = None

    # --- Keys by memorabilia type ---
    common_keys = [
        "title", "memorabilia_type", "personality", "autographed", "signed_by",
        "authentication", "year_of_issue", "condition", "dimensions", "materials",
        "limited_edition", "edition_number", "provenance", "manufacturer", 
        "ah_low_estimate_usd", "ah_high_estimate_usd", "description_block", "memorabilia_sale_report"
    ]

    sports_keys = [
        "sport", "league", "team", "player", "jersey_number",
        "game_used", "rookie_item", "match_event"
    ]

    auto_keys = [
        "driver", "series", "race_event", "race_used",
        "vehicle_model", "auto_memorabilia_type", "sponsor_logos"
    ]

    pop_keys = [
        "franchise", "character", "actor_or_creator", "media_type",
        "publisher_or_studio", "issue_number", "prop_used",
        "release_year", "genre"
    ]

    # Boolean fields with friendly labels
    boolean_labels = {
        "autographed": "Whether the item has an autograph",
        "limited_edition": "Whether the item is part of a limited edition",
        "game_used": "Whether the item was used in an official game",
        "rookie_item": "Whether the item is from the athlete’s rookie year",
        "race_used": "Whether the item was used during a race",
        "prop_used": "Whether the item was a screen-used or production-used prop",
    }

    # Select keys based on memorabilia_type
    mem_type_raw = (memorabilia.get("memorabilia_type") or "").strip().lower()
    if "sport" in mem_type_raw:
        allowed_keys = common_keys + sports_keys
    elif "auto" in mem_type_raw:
        allowed_keys = common_keys + auto_keys
    elif "pop" in mem_type_raw:
        allowed_keys = common_keys + pop_keys
    else:
        allowed_keys = common_keys  

    memorabilia_keys_prompt = []
    for key in allowed_keys:
        value = memorabilia.get(key)
        if value not in (None, "N/A", ""):
            if key in boolean_labels:
                formatted_key = boolean_labels[key]
                if isinstance(value, bool):
                    value = "Yes" if value else "No"
            else:
                formatted_key = key.replace("_", " ").title()
            memorabilia_keys_prompt.append(f"{formatted_key}: {value}")

    memorabilia_keys_prompt_str = "\n".join(memorabilia_keys_prompt)

    macro_prompt = memorabilia.get('macro_report', "No Information Available")

    # Get appraisals
    appraisal_str = memorabilia.get('appraisals')
    try:
        appraisals = ast.literal_eval(appraisal_str)
        
    except Exception as e:
        print("Appraisal error:", e)
        appraisals = []
    appraisals_block = ""
    skip_appraisal_keys = ["appraisal_ID", "openai_reasoning", "openai_confidence"]
    if isinstance(appraisals, list) and appraisals:    
        rendered = []
        for ap in appraisals:
            lines = []
            if ap["appraisal_type"].strip() == "Target Warranty":
                continue
            for k, v in ap.items(): 
                
                if k in skip_appraisal_keys:
                    continue
                
                # skip nullish/blank values
                if v is None:
                    continue
                if isinstance(v, str) and v.strip().lower() in na_values:
                    continue
    
                label = k.replace("_", " ").strip().capitalize()
                lines.append(f"{label}: {v}")
    
            if lines:
                rendered.append("\n".join(lines))
    
        if rendered:
            appraisals_block = "\n\nThird-party appraisals/estimates:\n" + "\n\n".join(rendered)

    
    memorabilia_prompt = f"""
Presented to you is the following memorabilia:
{memorabilia_keys_prompt_str}
{appraisals_block}

Macroeconomic Summary Report:
    {macro_prompt}
"""
    return memorabilia_prompt


def gpu_pricer_prompt_template(gpu):
    gpu = gpu.copy()

    na_values = {
        'n/a', 'na', 'none', '<n/a>', '<na>', 
        'nan', 'null', '<null>','missing', 
        'undef', 'undefined', 'not available', ''
    }
    
    for key, value in gpu.items():
        if isinstance(value, str):
            val_norm = value.strip().lower()
            if val_norm in na_values:
                gpu[key] = None
        elif pd.isna(value):
            gpu[key] = None

    allowed_keys = [
        "model_name", "brand", "gpu_model", "gpu_series", "gpu_architecture",
        "gpu_cores", "memory_size_gb", "memory_type", "memory_bandwidth_gbps",
        "pcie_version", "form_factor", "cooling_system_type", "base_clock_mhz",
        "boost_clock_mhz", "tdp_watts", "power_connectors", "outputs",
        "fabrication_process_nm", "die_size_mm2", "transistor_count_million",
        "release_date", "length", "width", "height", "dimension_unit",
        "weight", "weight_unit", 
        "lot_url", "ah_bp_price_usd", "ah_price_usd",
        #  "ah_low_estimate_usd", "ah_high_estimate_usd", 
        "description_block", "gpu_sale_report"
    ]

    gpu_keys_prompt = []
    for key in allowed_keys:
        value = gpu.get(key)
        if value not in (None, "N/A", ""):
            formatted_key = key.replace("_", " ").title()
            if key == "description_block":
                # Remove whitespace and newlines, enclose in triple quotes
                cleaned_value = " ".join(str(value).split())
                gpu_keys_prompt.append(f'{formatted_key}: """{cleaned_value}"""')
            else:
                gpu_keys_prompt.append(f"{formatted_key}: {value}")

    gpu_keys_prompt_str = "\n".join(gpu_keys_prompt)

    macro_prompt = gpu.get('macro_report', "No Information Available")
    
    gpu_prompt = f"""
Presented to you is the following gpu:
{gpu_keys_prompt_str}

Macroeconomic Summary Report:
    {macro_prompt}
"""
    return gpu_prompt


