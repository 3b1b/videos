"""
Geographic data for the 630 contestants at the 2025 International
Mathematical Olympiad, held in Sunshine Coast, Australia.

Contestant counts are from https://www.imo-official.org/results/individual/year/2025/

Each entry is (country, num_participants, latitude, longitude, color), where
the coordinates mark the city the team most plausibly departed from, so that
lines drawn toward Australia approximate the actual journeys. Two are the
confirmed sites of the 2025 pre-IMO training camps:

    United States  MOP at the Illinois Mathematics and Science Academy in
                   Aurora, Illinois, so departing Chicago
    Canada         camp at the Banff International Research Station, running
                   until July 11th, so departing Calgary

Every other entry is that country's principal international gateway airport,
which is usually in the capital or largest city but sometimes not:

    Bhutan         Paro, the country's only international airport
    Cyprus         Larnaca, since Nicosia has no civilian airport
    Liechtenstein  Zurich, since the country has no airport
    Palestine      Amman, Jordan, the usual gateway for West Bank travellers
    Ukraine        Warsaw, since Ukrainian airspace was closed to civil
                   aviation throughout 2025

These are informed guesses rather than researched itineraries; a country whose
line matters on screen is worth checking individually.

The color is the one most associated with the country, almost always its
dominant flag color, with the source noted in the comment. Flags being what
they are, plenty of these land on nearly the same red, so colors distinguish a
line's origin only loosely.
"""

# Where the 2025 IMO was held
HOST_NAME = "Sunshine Coast, Australia"
HOST_LATITUDE = -26.6500
HOST_LONGITUDE = 153.0667

# (country, num_participants, latitude, longitude, color)
IMO_2025_PARTICIPANTS = [
    ('Albania'                   , 6,  41.4147,   19.7206, '#E41E20'),  # Tirana (TIA)
    ('Argentina'                 , 6, -34.8222,  -58.5358, '#74ACDF'),  # Buenos Aires (EZE)
    ('Armenia'                   , 6,  40.1473,   44.3959, '#F2A800'),  # Yerevan (EVN)
    ('Australia'                 , 6, -33.9399,  151.1753, '#FFCD00'),  # Sydney (SYD)
    ('Austria'                   , 6,  48.1103,   16.5697, '#ED2939'),  # Vienna (VIE)
    ('Azerbaijan'                , 6,  40.4675,   50.0467, '#00B5E2'),  # Baku (GYD)
    ('Bangladesh'                , 6,  23.8433,   90.3978, '#006A4E'),  # Dhaka (DAC)
    ('Belarus'                   , 6,  53.8825,   28.0307, '#CE1720'),  # Minsk (MSQ)
    ('Belgium'                   , 6,  50.9014,    4.4844, '#FDDA24'),  # Brussels (BRU)
    ('Bhutan'                    , 6,  27.4032,   89.4246, '#FF6E00'),  # Paro (PBH)
    ('Bosnia and Herzegovina'    , 6,  43.8246,   18.3315, '#1B4F9C'),  # Sarajevo (SJJ)
    ('Botswana'                  , 6, -24.5552,   25.9182, '#75AADB'),  # Gaborone (GBE)
    ('Brazil'                    , 6, -23.4356,  -46.4731, '#009739'),  # Sao Paulo (GRU)
    ('Bulgaria'                  , 6,  42.6952,   23.4062, '#00966E'),  # Sofia (SOF)
    ('Cambodia'                  , 6,  11.5466,  104.8441, '#1B4CA0'),  # Phnom Penh (PNH)
    ('Canada'                    , 6,  51.1315, -114.0106, '#D80621'),  # Calgary (YYC)  <- training camp
    ('Colombia'                  , 6,   4.7016,  -74.1469, '#FCD116'),  # Bogota (BOG)
    ('Costa Rica'                , 6,   9.9939,  -84.2088, '#0033A0'),  # San Jose (SJO)
    ('Croatia'                   , 6,  45.7429,   16.0688, '#E62B2B'),  # Zagreb (ZAG)
    ('Cuba'                      , 6,  22.9892,  -82.4091, '#002A8F'),  # Havana (HAV)
    ('Cyprus'                    , 6,  34.8751,   33.6249, '#D57800'),  # Larnaca (LCA)
    ('Czech Republic'            , 6,  50.1008,   14.2600, '#11457E'),  # Prague (PRG)
    ('Denmark'                   , 6,  55.6180,   12.6508, '#C8102E'),  # Copenhagen (CPH)
    ('Dominican Republic'        , 6,  18.4297,  -69.6689, '#002D62'),  # Santo Domingo (SDQ)
    ('Ecuador'                   , 6,  -0.1292,  -78.3575, '#FFDD00'),  # Quito (UIO)
    ('Estonia'                   , 6,  59.4133,   24.8328, '#0072CE'),  # Tallinn (TLL)
    ('Finland'                   , 6,  60.3172,   24.9633, '#0053A5'),  # Helsinki (HEL)
    ('France'                    , 6,  49.0097,    2.5479, '#0055A4'),  # Paris (CDG)
    ('Georgia'                   , 6,  41.6692,   44.9547, '#DA291C'),  # Tbilisi (TBS)
    ('Germany'                   , 6,  50.0379,    8.5622, '#FFCE00'),  # Frankfurt (FRA)
    ('Greece'                    , 6,  37.9364,   23.9445, '#0D5EAF'),  # Athens (ATH)
    ('Honduras'                  , 6,  14.3608,  -87.6217, '#0073CF'),  # Tegucigalpa (XPL)
    ('Hong Kong'                 , 6,  22.3080,  113.9185, '#DE2910'),  # Hong Kong (HKG)
    ('Hungary'                   , 6,  47.4369,   19.2556, '#436F4D'),  # Budapest (BUD)
    ('Iceland'                   , 6,  63.9850,  -22.6056, '#02529C'),  # Keflavik (KEF)
    ('India'                     , 6,  19.0896,   72.8656, '#FF9933'),  # Mumbai (BOM)
    ('Indonesia'                 , 6,  -6.1256,  106.6559, '#CE1126'),  # Jakarta (CGK)
    ('Ireland'                   , 6,  53.4213,   -6.2701, '#169B62'),  # Dublin (DUB)
    ('Islamic Republic of Iran'  , 6,  35.4161,   51.1522, '#239F40'),  # Tehran (IKA)
    ('Israel'                    , 6,  32.0114,   34.8867, '#0038B8'),  # Tel Aviv (TLV)
    ('Italy'                     , 6,  41.8003,   12.2389, '#008C45'),  # Rome (FCO)
    ('Ivory Coast'               , 6,   5.2614,   -3.9263, '#F77F00'),  # Abidjan (ABJ)
    ('Japan'                     , 6,  35.7720,  140.3929, '#BC002D'),  # Tokyo (NRT)
    ('Kazakhstan'                , 6,  43.3521,   77.0405, '#00AFCA'),  # Almaty (ALA)
    ('Kenya'                     , 6,  -1.3192,   36.9278, '#BE0027'),  # Nairobi (NBO)
    ('Kosovo'                    , 6,  42.5728,   21.0358, '#244AA5'),  # Pristina (PRN)
    ('Kyrgyzstan'                , 6,  43.0613,   74.4776, '#E8112D'),  # Bishkek (FRU)
    ('Latvia'                    , 6,  56.9236,   23.9711, '#9E3039'),  # Riga (RIX)
    ('Lithuania'                 , 6,  54.6341,   25.2858, '#FDB913'),  # Vilnius (VNO)
    ('Macau'                     , 6,  22.1496,  113.5915, '#00785E'),  # Macau (MFM)
    ('Malaysia'                  , 6,   2.7456,  101.7099, '#0032A0'),  # Kuala Lumpur (KUL)
    ('Mexico'                    , 6,  19.4363,  -99.0721, '#006847'),  # Mexico City (MEX)
    ('Mongolia'                  , 6,  47.6533,  106.8197, '#DA2032'),  # Ulaanbaatar (UBN)
    ('Montenegro'                , 6,  42.3594,   19.2519, '#D4AF37'),  # Podgorica (TGD)
    ('Morocco'                   , 6,  33.3675,   -7.5900, '#C1272D'),  # Casablanca (CMN)
    ('Myanmar'                   , 6,  16.9073,   96.1332, '#FECB00'),  # Yangon (RGN)
    ('Nepal'                     , 6,  27.6966,   85.3591, '#DC143C'),  # Kathmandu (KTM)
    ('Netherlands'               , 6,  52.3105,    4.7683, '#FF6C00'),  # Amsterdam (AMS)
    ('New Zealand'               , 6, -37.0082,  174.7850, '#B0B7BC'),  # Auckland (AKL)
    ('North Macedonia'           , 6,  41.9616,   21.6214, '#F8E600'),  # Skopje (SKP)
    ('Norway'                    , 6,  60.1976,   11.1004, '#BA0C2F'),  # Oslo (OSL)
    ('Pakistan'                  , 6,  33.5490,   72.8256, '#046A38'),  # Islamabad (ISB)
    ("People's Republic of China", 6,  40.0799,  116.6031, '#DE2910'),  # Beijing (PEK)
    ('Peru'                      , 6, -12.0219,  -77.1143, '#D91023'),  # Lima (LIM)
    ('Philippines'               , 6,  14.5086,  121.0198, '#0038A8'),  # Manila (MNL)
    ('Poland'                    , 6,  52.1657,   20.9671, '#D4213D'),  # Warsaw (WAW)
    ('Portugal'                  , 6,  38.7742,   -9.1342, '#00843D'),  # Lisbon (LIS)
    ('Qatar'                     , 6,  25.2731,   51.6081, '#8A1538'),  # Doha (DOH)
    ('Republic of Korea'         , 6,  37.4602,  126.4407, '#0047A0'),  # Seoul (ICN)
    ('Republic of Moldova'       , 6,  46.9277,   28.9309, '#0046AE'),  # Chisinau (KIV)
    ('Romania'                   , 6,  44.5711,   26.0850, '#002B7F'),  # Bucharest (OTP)
    ('Russia'                    , 6,  55.9726,   37.4146, '#0039A6'),  # Moscow (SVO)
    ('Rwanda'                    , 6,  -1.9686,   30.1395, '#20603D'),  # Kigali (KGL)
    ('Saudi Arabia'              , 6,  24.9576,   46.6988, '#006C35'),  # Riyadh (RUH)
    ('Serbia'                    , 6,  44.8184,   20.3091, '#C6363C'),  # Belgrade (BEG)
    ('Singapore'                 , 6,   1.3644,  103.9915, '#EF3340'),  # Singapore (SIN)
    ('Slovakia'                  , 6,  48.1702,   17.2127, '#0B4EA2'),  # Bratislava (BTS)
    ('Slovenia'                  , 6,  46.2237,   14.4576, '#0F52BA'),  # Ljubljana (LJU)
    ('South Africa'              , 6, -26.1367,   28.2411, '#007A4D'),  # Johannesburg (JNB)
    ('Spain'                     , 6,  40.4936,   -3.5668, '#AA151B'),  # Madrid (MAD)
    ('Sri Lanka'                 , 6,   7.1808,   79.8841, '#8D153A'),  # Colombo (CMB)
    ('Sweden'                    , 6,  59.6519,   17.9186, '#FECC02'),  # Stockholm (ARN)
    ('Switzerland'               , 6,  47.4647,    8.5492, '#FF0000'),  # Zurich (ZRH)
    ('Syria'                     , 6,  33.4114,   36.5156, '#009B3A'),  # Damascus (DAM)
    ('Taiwan'                    , 6,  25.0777,  121.2328, '#0B318F'),  # Taipei (TPE)
    ('Tajikistan'                , 6,  38.5433,   68.8250, '#CC0000'),  # Dushanbe (DYU)
    ('Thailand'                  , 6,  13.6900,  100.7501, '#2D2A4A'),  # Bangkok (BKK)
    ('Tunisia'                   , 6,  36.8510,   10.2272, '#E70013'),  # Tunis (TUN)
    ('Turkmenistan'              , 6,  37.9868,   58.3610, '#00843D'),  # Ashgabat (ASB)
    ('Türkiye'                   , 6,  41.2753,   28.7519, '#E30A17'),  # Istanbul (IST)
    ('Uganda'                    , 6,   0.0424,   32.4435, '#FCDC04'),  # Entebbe (EBB)
    ('Ukraine'                   , 6,  52.1657,   20.9671, '#0057B7'),  # Warsaw, Poland (WAW)
    ('United Arab Emirates'      , 6,  25.2532,   55.3657, '#007A33'),  # Dubai (DXB)
    ('United Kingdom'            , 6,  51.4700,   -0.4543, '#012169'),  # London (LHR)
    ('United States of America'  , 6,  41.9742,  -87.9073, '#29ABCA'),  # Chicago (ORD)  <- training camp
    ('Uzbekistan'                , 6,  41.2579,   69.2812, '#0099B5'),  # Tashkent (TAS)
    ('Vietnam'                   , 6,  21.2212,  105.8072, '#DA251D'),  # Hanoi (HAN)
    ('Algeria'                   , 5,  36.6910,    3.2154, '#007229'),  # Algiers (ALG)
    ('Iraq'                      , 5,  33.2625,   44.2346, '#007A3D'),  # Baghdad (BGW)
    ('Palestine'                 , 5,  31.7226,   35.9932, '#009639'),  # Amman, Jordan (AMM)
    ('Uruguay'                   , 5, -34.8384,  -56.0308, '#7BAFD4'),  # Montevideo (MVD)
    ('Bolivia'                   , 4, -17.6448,  -63.1354, '#007A33'),  # Santa Cruz de la Sierra (VVI)
    ('Namibia'                   , 4, -22.4799,   17.4709, '#0033A0'),  # Windhoek (WDH)
    ('Paraguay'                  , 4, -25.2400,  -57.5200, '#D52B1E'),  # Asuncion (ASU)
    ('Chile'                     , 3, -33.3930,  -70.7858, '#0039A6'),  # Santiago (SCL)
    ('El Salvador'               , 3,  13.4409,  -89.0557, '#0F47AF'),  # San Salvador (SAL)
    ('Luxembourg'                , 3,  49.6266,    6.2115, '#00A1DE'),  # Luxembourg (LUX)
    ('Puerto Rico'               , 3,  18.4394,  -66.0018, '#41B6E6'),  # San Juan (SJU)
    ('Ghana'                     , 2,   5.6052,   -0.1668, '#FCD20F'),  # Accra (ACC)
    ('Cameroon'                  , 1,   4.0061,    9.7195, '#007A5E'),  # Douala (DLA)
    ('Liechtenstein'             , 1,  47.4647,    8.5492, '#002B7F'),  # Zurich (ZRH)
]

TOTAL_PARTICIPANTS = 630  # across 111 countries
