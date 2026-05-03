// =============================================================================
// labFields.ts — MIMIC-IV Lab Field Configuration
// Derived from code/constants.py LABS dictionary (50 lab features).
// Used by NewCaseWizard for dynamic form generation and bulk upload parsing.
// =============================================================================

export interface LabFieldDef {
  /** MIMIC-IV itemid (string key used in CSV columns & labs_means.json) */
  itemid: string;
  /** Human-readable lab name */
  name: string;
  /** Measurement unit */
  unit: string;
  /** Normal range string for display */
  normalRange: string;
  /** Clinical category for grouped rendering */
  category: LabCategory;
}

export type LabCategory =
  | 'CBC'
  | 'Metabolic'
  | 'Coagulation'
  | 'Liver'
  | 'ABG'
  | 'Differential'
  | 'Other';

export const LAB_CATEGORY_LABELS: Record<LabCategory, string> = {
  CBC: 'Complete Blood Count (CBC)',
  Metabolic: 'Basic Metabolic Panel',
  Coagulation: 'Coagulation Studies',
  Liver: 'Liver Function Tests',
  ABG: 'Arterial Blood Gas',
  Differential: 'WBC Differential',
  Other: 'Other Labs',
};

/**
 * All 50 MIMIC-IV lab features used by the Symile-MIMIC model.
 * Order matches constants.py LABS dictionary.
 */
export const LAB_FIELDS: LabFieldDef[] = [
  // ── CBC ──────────────────────────────────────────────────
  { itemid: '51221', name: 'Hematocrit',      unit: '%',       normalRange: '36–46',     category: 'CBC' },
  { itemid: '51265', name: 'Platelet Count',   unit: 'K/µL',   normalRange: '150–400',   category: 'CBC' },
  { itemid: '51222', name: 'Hemoglobin',       unit: 'g/dL',   normalRange: '12–17.5',   category: 'CBC' },
  { itemid: '51301', name: 'White Blood Cells', unit: 'K/µL',  normalRange: '4.5–11',    category: 'CBC' },
  { itemid: '51279', name: 'Red Blood Cells',  unit: 'M/µL',   normalRange: '4.2–6.1',   category: 'CBC' },
  { itemid: '51249', name: 'MCHC',             unit: 'g/dL',   normalRange: '31–37',     category: 'CBC' },
  { itemid: '51250', name: 'MCV',              unit: 'fL',     normalRange: '80–100',    category: 'CBC' },
  { itemid: '51248', name: 'MCH',              unit: 'pg',     normalRange: '27–33',     category: 'CBC' },
  { itemid: '51277', name: 'RDW',              unit: '%',      normalRange: '11.5–14.5', category: 'CBC' },
  { itemid: '52172', name: 'RDW-SD',           unit: 'fL',     normalRange: '39–46',     category: 'CBC' },

  // ── Metabolic ───────────────────────────────────────────
  { itemid: '50912', name: 'Creatinine',       unit: 'mg/dL',  normalRange: '0.7–1.3',   category: 'Metabolic' },
  { itemid: '50971', name: 'Potassium',        unit: 'mEq/L',  normalRange: '3.5–5.0',   category: 'Metabolic' },
  { itemid: '51006', name: 'Urea Nitrogen',    unit: 'mg/dL',  normalRange: '7–20',      category: 'Metabolic' },
  { itemid: '50983', name: 'Sodium',           unit: 'mEq/L',  normalRange: '136–145',   category: 'Metabolic' },
  { itemid: '50902', name: 'Chloride',         unit: 'mEq/L',  normalRange: '98–106',    category: 'Metabolic' },
  { itemid: '50882', name: 'Bicarbonate',      unit: 'mEq/L',  normalRange: '22–29',     category: 'Metabolic' },
  { itemid: '50868', name: 'Anion Gap',        unit: 'mEq/L',  normalRange: '3–11',      category: 'Metabolic' },
  { itemid: '50931', name: 'Glucose',          unit: 'mg/dL',  normalRange: '70–100',    category: 'Metabolic' },
  { itemid: '50960', name: 'Magnesium',        unit: 'mg/dL',  normalRange: '1.7–2.2',   category: 'Metabolic' },
  { itemid: '50893', name: 'Calcium, Total',   unit: 'mg/dL',  normalRange: '8.5–10.5',  category: 'Metabolic' },
  { itemid: '50970', name: 'Phosphate',        unit: 'mg/dL',  normalRange: '2.5–4.5',   category: 'Metabolic' },
  { itemid: '50813', name: 'Lactate',          unit: 'mmol/L', normalRange: '0.5–2.0',   category: 'Metabolic' },

  // ── Coagulation ─────────────────────────────────────────
  { itemid: '51237', name: 'INR(PT)',          unit: '',       normalRange: '0.8–1.1',   category: 'Coagulation' },
  { itemid: '51274', name: 'PT',               unit: 'sec',    normalRange: '11–13.5',   category: 'Coagulation' },
  { itemid: '51275', name: 'PTT',              unit: 'sec',    normalRange: '25–35',     category: 'Coagulation' },

  // ── Liver ───────────────────────────────────────────────
  { itemid: '50861', name: 'ALT',              unit: 'IU/L',   normalRange: '7–56',      category: 'Liver' },
  { itemid: '50878', name: 'AST',              unit: 'IU/L',   normalRange: '10–40',     category: 'Liver' },
  { itemid: '50863', name: 'Alkaline Phosphatase', unit: 'IU/L', normalRange: '44–147', category: 'Liver' },
  { itemid: '50885', name: 'Bilirubin, Total', unit: 'mg/dL',  normalRange: '0.1–1.2',   category: 'Liver' },
  { itemid: '50862', name: 'Albumin',          unit: 'g/dL',   normalRange: '3.4–5.4',   category: 'Liver' },

  // ── ABG ─────────────────────────────────────────────────
  { itemid: '50820', name: 'pH',               unit: '',       normalRange: '7.35–7.45', category: 'ABG' },
  { itemid: '50821', name: 'pO2',              unit: 'mmHg',   normalRange: '75–100',    category: 'ABG' },
  { itemid: '50818', name: 'pCO2',             unit: 'mmHg',   normalRange: '35–45',     category: 'ABG' },
  { itemid: '50802', name: 'Base Excess',      unit: 'mEq/L',  normalRange: '-2 to +2',  category: 'ABG' },
  { itemid: '50804', name: 'Calc Total CO2',   unit: 'mEq/L',  normalRange: '23–29',     category: 'ABG' },

  // ── Differential ────────────────────────────────────────
  { itemid: '51256', name: 'Neutrophils',      unit: '%',      normalRange: '40–70',     category: 'Differential' },
  { itemid: '51244', name: 'Lymphocytes',      unit: '%',      normalRange: '20–40',     category: 'Differential' },
  { itemid: '51254', name: 'Monocytes',        unit: '%',      normalRange: '2–8',       category: 'Differential' },
  { itemid: '51200', name: 'Eosinophils',      unit: '%',      normalRange: '1–4',       category: 'Differential' },
  { itemid: '51146', name: 'Basophils',        unit: '%',      normalRange: '0–1',       category: 'Differential' },
  { itemid: '52075', name: 'Abs Neutrophil Count', unit: 'K/µL', normalRange: '1.5–8', category: 'Differential' },
  { itemid: '51133', name: 'Abs Lymphocyte Count', unit: 'K/µL', normalRange: '1.0–4.0', category: 'Differential' },
  { itemid: '52074', name: 'Abs Monocyte Count', unit: 'K/µL', normalRange: '0.2–1.0',  category: 'Differential' },
  { itemid: '52073', name: 'Abs Eosinophil Count', unit: 'K/µL', normalRange: '0–0.5',  category: 'Differential' },
  { itemid: '52069', name: 'Abs Basophil Count', unit: 'K/µL', normalRange: '0–0.2',    category: 'Differential' },

  // ── Other ───────────────────────────────────────────────
  { itemid: '50934', name: 'H',                unit: '',       normalRange: '—',         category: 'Other' },
  { itemid: '51678', name: 'L',                unit: '',       normalRange: '—',         category: 'Other' },
  { itemid: '50947', name: 'I',                unit: '',       normalRange: '—',         category: 'Other' },
  { itemid: '50910', name: 'Creatine Kinase (CK)', unit: 'IU/L', normalRange: '22–198', category: 'Other' },
  { itemid: '52135', name: 'Immature Granulocytes', unit: '%', normalRange: '0–0.5',    category: 'Other' },
];

/** Quick lookup: itemid → LabFieldDef */
export const LAB_FIELD_MAP = new Map(LAB_FIELDS.map(f => [f.itemid, f]));

/** Quick lookup: lab name (case-insensitive) → LabFieldDef */
export const LAB_NAME_MAP = new Map(LAB_FIELDS.map(f => [f.name.toLowerCase(), f]));

/** All unique categories in display order */
export const LAB_CATEGORIES: LabCategory[] = [
  'CBC', 'Metabolic', 'Coagulation', 'Liver', 'ABG', 'Differential', 'Other',
];

/** Type for lab values form state: itemid → numeric value or null */
export type LabValues = Record<string, number | null>;

/**
 * Parse a CSV string (e.g. from a bulk upload) into a LabValues object.
 * Supports columns named by itemid ("51221") or lab name ("Hematocrit").
 */
export function parseLabCSV(csvText: string): LabValues {
  const lines = csvText.trim().split('\n');
  if (lines.length < 2) return {};

  const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
  const values = lines[1].split(',').map(v => v.trim().replace(/"/g, ''));

  const result: LabValues = {};

  headers.forEach((header, idx) => {
    const val = parseFloat(values[idx]);
    if (isNaN(val)) return;

    // Try matching by itemid directly
    if (LAB_FIELD_MAP.has(header)) {
      result[header] = val;
      return;
    }

    // Try matching by lab name (case-insensitive)
    const byName = LAB_NAME_MAP.get(header.toLowerCase());
    if (byName) {
      result[byName.itemid] = val;
    }
  });

  return result;
}

/**
 * Parse a JSON string (e.g. from a bulk upload) into a LabValues object.
 * Keys can be itemids or lab names.
 */
export function parseLabJSON(jsonText: string): LabValues {
  const obj = JSON.parse(jsonText);
  const result: LabValues = {};

  for (const [key, value] of Object.entries(obj)) {
    const numVal = typeof value === 'number' ? value : parseFloat(value as string);
    if (isNaN(numVal)) continue;

    if (LAB_FIELD_MAP.has(key)) {
      result[key] = numVal;
      continue;
    }

    const byName = LAB_NAME_MAP.get(key.toLowerCase());
    if (byName) {
      result[byName.itemid] = numVal;
    }
  }

  return result;
}
