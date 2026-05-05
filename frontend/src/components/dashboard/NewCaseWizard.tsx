'use client';

import React, { useState, useEffect, useRef } from 'react';
import { X, UploadCloud, ChevronRight, ChevronLeft, CheckCircle, UserPlus, FileText, Activity, AlertCircle, Search, ChevronDown, ChevronUp } from 'lucide-react';
import { useCaseStore } from '@/store/caseStore';
import { createCase } from '@/lib/api';
import { useToastStore } from '@/store/toastStore';
import { LAB_FIELDS, LAB_CATEGORIES, LAB_CATEGORY_LABELS, parseLabCSV, parseLabJSON, type LabValues, type LabCategory } from '@/lib/labFields';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return debouncedValue;
}

interface NewCaseWizardProps {
  isOpen: boolean;
  onClose: () => void;
}

type StepErrors = Record<string, string>;

export function NewCaseWizard({ isOpen, onClose }: NewCaseWizardProps) {
  const addCase = useCaseStore((state) => state.addCase);
  const addToast = useToastStore((state) => state.addToast);

  const [step, setStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errors, setErrors] = useState<StepErrors>({});
  const [hasStepErrors, setHasStepErrors] = useState(false);

  // Form State
  const [formData, setFormData] = useState({
    firstName: '', lastName: '', mrn: '', age: '', sex: 'M',
    rhythm: 'Normal Sinus Rhythm',
    mimicSubjectId: '',
  });

  // Lab values — keyed by MIMIC-IV itemid
  const [labValues, setLabValues] = useState<LabValues>({});
  const [labSearch, setLabSearch] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<LabCategory>>(new Set(['CBC', 'Metabolic']));

  const [imageFile, setImageFile] = useState<File | null>(null);

  // Debounced troponin for live signal (itemid 50947 is "I" but we don't have troponin in MIMIC)
  // Use Creatinine (50912) as a proxy risk marker for demo
  const filledLabCount = Object.values(labValues).filter(v => v !== null && v !== undefined).length;

  useEffect(() => {
    if (!isOpen) {
      setTimeout(() => {
        setStep(1); setIsSubmitting(false); setErrors({});
        setFormData({ firstName: '', lastName: '', mrn: '', age: '', sex: 'M', rhythm: 'Normal Sinus Rhythm', mimicSubjectId: '' });
        setLabValues({}); setLabSearch(''); setImageFile(null);
        setExpandedCategories(new Set(['CBC', 'Metabolic']));
      }, 300);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  // ── Validation ──────────────────────────────────────────

  const validateStep = (s: number): boolean => {
    const newErrors: StepErrors = {};
    if (s === 1) {
      if (!formData.firstName.trim()) newErrors.firstName = 'First name is required';
      if (!formData.lastName.trim()) newErrors.lastName = 'Last name is required';
      if (!formData.mrn.trim()) newErrors.mrn = 'MRN is required';
      if (!formData.age.trim() || isNaN(Number(formData.age)) || Number(formData.age) <= 0)
        newErrors.age = 'Valid age is required';
    }
    // Step 2: labs are optional (model imputes missing values)
    // Step 3: image is optional
    setErrors(newErrors);
    const hasErrors = Object.keys(newErrors).length > 0;
    setHasStepErrors(hasErrors);
    return !hasErrors;
  };

  const handleNext = () => {
    if (validateStep(step)) {
      setHasStepErrors(false);
      setStep((s) => Math.min(s + 1, 4));
    }
  };
  const handlePrev = () => { setErrors({}); setStep((s) => Math.max(s - 1, 1)); };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    if (errors[name]) setErrors(prev => { const n = { ...prev }; delete n[name]; return n; });
  };

  const handleLabChange = (itemid: string, value: string) => {
    setLabValues(prev => ({
      ...prev,
      [itemid]: value === '' ? null : parseFloat(value),
    }));
  };

  // ── Bulk Upload ─────────────────────────────────────────
  const handleBulkUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const text = ev.target?.result as string;
        let parsed: LabValues;
        if (file.name.endsWith('.json')) {
          parsed = parseLabJSON(text);
        } else {
          parsed = parseLabCSV(text);
        }
        const count = Object.keys(parsed).length;
        setLabValues(prev => ({ ...prev, ...parsed }));
        addToast({ type: 'success', title: 'Labs Imported', message: `${count} lab values loaded from ${file.name}` });
      } catch {
        addToast({ type: 'error', title: 'Parse Error', message: 'Could not parse the uploaded file. Use CSV or JSON format.' });
      }
    };
    reader.readAsText(file);
  };

  // ── Image Upload ────────────────────────────────────────
  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files?.[0]) setImageFile(e.dataTransfer.files[0]);
  };
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setImageFile(e.target.files[0]);
  };

  // ── Submit ──────────────────────────────────────────────
  const handleSubmit = async () => {
    setIsSubmitting(true);
    try {
      const trimmedSubject = formData.mimicSubjectId.trim();
      const payload: Record<string, unknown> = { ...formData, labs: labValues };
      if (trimmedSubject) {
        const parsedSubject = Number.parseInt(trimmedSubject, 10);
        if (Number.isFinite(parsedSubject) && parsedSubject > 0) {
          payload.mimic_subject_id = parsedSubject;
        }
      }
      const data = new FormData();
      data.append('case_data', JSON.stringify(payload));
      if (imageFile) data.append('image', imageFile);

      const newCaseSummary = await createCase(data);
      addCase(newCaseSummary);
      addToast({ type: 'success', title: 'Patient Registered', message: 'Case ingested into the platform.' });
      onClose();
    } catch (err: any) {
      addToast({ type: 'error', title: 'Ingestion Failed', message: err.message || 'An error occurred.' });
    } finally {
      setIsSubmitting(false);
    }
  };

  // ── Lab filtering ───────────────────────────────────────
  const filteredFields = labSearch
    ? LAB_FIELDS.filter(f => f.name.toLowerCase().includes(labSearch.toLowerCase()) || f.itemid.includes(labSearch))
    : LAB_FIELDS;

  const toggleCategory = (cat: LabCategory) => {
    setExpandedCategories(prev => {
      const next = new Set(prev);
      next.has(cat) ? next.delete(cat) : next.add(cat);
      return next;
    });
  };

  const inputClass = (field: string) => cn(
    "w-full border rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500 outline-none text-sm",
    errors[field] ? "border-red-400 bg-red-50" : "border-gray-300"
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl overflow-hidden flex flex-col max-h-[90vh]">

        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between bg-gray-50/50">
          <div className="flex items-center gap-2 text-blue-600">
            <UserPlus className="w-5 h-5" />
            <h2 className="text-lg font-bold text-gray-900">Clinical Ingestion Wizard</h2>
          </div>
          <button onClick={onClose} className="p-2 text-gray-400 hover:bg-gray-100 rounded-full transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Progress Bar */}
        <div className="px-6 py-4 border-b border-gray-100 bg-white">
          <div className="flex items-center justify-between mb-2">
            {['Demographics', 'Lab Data', 'Imaging', 'Review'].map((label, idx) => (
              <div key={label} className={cn("text-xs font-semibold uppercase tracking-wider", step >= idx + 1 ? "text-blue-600" : "text-gray-400")}>
                Step {idx + 1}
              </div>
            ))}
          </div>
          <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
            <div className="h-full bg-blue-600 transition-all duration-300" style={{ width: `${((step - 1) / 3) * 100}%` }} />
          </div>
        </div>

        {/* Body */}
        <div className="p-6 overflow-y-auto flex-1">

          {/* STEP 1: Demographics */}
          {step === 1 && (
            <div className="space-y-5 animate-in slide-in-from-right-4 fade-in duration-300">
              <h3 className="text-xl font-bold text-gray-800 mb-2">Patient Demographics</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">First Name <span className="text-red-500">*</span></label>
                  <input type="text" name="firstName" value={formData.firstName} onChange={handleChange} className={inputClass('firstName')} placeholder="Jane" />
                  {errors.firstName && <p className="text-xs text-red-500 mt-1 flex items-center gap-1"><AlertCircle className="w-3 h-3" />{errors.firstName}</p>}
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Last Name <span className="text-red-500">*</span></label>
                  <input type="text" name="lastName" value={formData.lastName} onChange={handleChange} className={inputClass('lastName')} placeholder="Doe" />
                  {errors.lastName && <p className="text-xs text-red-500 mt-1 flex items-center gap-1"><AlertCircle className="w-3 h-3" />{errors.lastName}</p>}
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">MRN <span className="text-red-500">*</span></label>
                  <input type="text" name="mrn" value={formData.mrn} onChange={handleChange} className={inputClass('mrn')} placeholder="MRN-12345" />
                  {errors.mrn && <p className="text-xs text-red-500 mt-1 flex items-center gap-1"><AlertCircle className="w-3 h-3" />{errors.mrn}</p>}
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Age <span className="text-red-500">*</span></label>
                  <input type="number" name="age" value={formData.age} onChange={handleChange} className={inputClass('age')} placeholder="65" />
                  {errors.age && <p className="text-xs text-red-500 mt-1 flex items-center gap-1"><AlertCircle className="w-3 h-3" />{errors.age}</p>}
                </div>
                <div className="col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Biological Sex</label>
                  <select name="sex" value={formData.sex} onChange={handleChange} className="w-full border-gray-300 rounded-lg p-2.5 border focus:ring-2 focus:ring-blue-500 outline-none bg-white text-sm">
                    <option value="M">Male</option>
                    <option value="F">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
                <div className="col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-1 flex items-center gap-2">
                    Reference MIMIC subject_id
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-violet-50 text-violet-700 border border-violet-200">Optional</span>
                  </label>
                  <input
                    type="number"
                    name="mimicSubjectId"
                    value={formData.mimicSubjectId}
                    onChange={handleChange}
                    className={inputClass('mimicSubjectId')}
                    placeholder="e.g. 10011938"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Linking a MIMIC-IV subject pulls their real CXR + ECG + Labs tensors into the
                    Symile multimodal index, enabling cross-modal retrieval against the 38 backfilled cases.
                    Leave blank to register a synthetic case (CXR-only retrieval).
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* STEP 2: Lab Data — Dynamic from MIMIC-IV schema */}
          {step === 2 && (
            <div className="space-y-4 animate-in slide-in-from-right-4 fade-in duration-300">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xl font-bold text-gray-800">Laboratory Data</h3>
                  <p className="text-xs text-gray-500 mt-1">{filledLabCount} of {LAB_FIELDS.length} labs entered • Missing values will be imputed by the model</p>
                </div>
              </div>

              {/* Bulk Upload Dropzone */}
              <div
                className={cn(
                  "border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center text-center transition-colors cursor-pointer",
                  filledLabCount > 0 ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-gray-400 hover:bg-gray-50"
                )}
                onDragOver={e => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const file = e.dataTransfer.files?.[0];
                  if (file) {
                    const fakeEvent = { target: { files: [file] } } as unknown as React.ChangeEvent<HTMLInputElement>;
                    handleBulkUpload(fakeEvent);
                  }
                }}
                onClick={() => document.getElementById('lab-upload')?.click()}
              >
                <input id="lab-upload" type="file" className="hidden" accept=".csv,.json" onChange={handleBulkUpload} />
                <FileText className={cn("w-10 h-10 mb-3", filledLabCount > 0 ? "text-blue-500" : "text-gray-400")} />
                {filledLabCount > 0 ? (
                  <>
                    <p className="text-blue-900 font-semibold">{filledLabCount} labs imported successfully.</p>
                    <p className="text-blue-600 text-sm mt-1">You can upload again to replace data.</p>
                  </>
                ) : (
                  <>
                    <p className="text-gray-700 font-medium">Click or drag & drop to bulk upload Lab & ECG data</p>
                    <p className="text-gray-500 text-sm mt-1">Accepts CSV or JSON formats</p>
                  </>
                )}
              </div>

              {/* Search */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text" placeholder="Search labs..."
                  value={labSearch} onChange={e => setLabSearch(e.target.value)}
                  className="w-full pl-9 pr-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                />
              </div>

              {/* ECG Rhythm */}
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Activity className="w-4 h-4 text-blue-600" />
                  <span className="text-sm font-semibold text-gray-800">ECG Rhythm</span>
                </div>
                <select name="rhythm" value={formData.rhythm} onChange={handleChange} className="w-full border-gray-300 rounded-lg p-2 border text-sm outline-none bg-white">
                  <option value="Normal Sinus Rhythm">Normal Sinus Rhythm</option>
                  <option value="Atrial Fibrillation">Atrial Fibrillation</option>
                  <option value="Sinus Tachycardia">Sinus Tachycardia</option>
                </select>
              </div>

              {/* Grouped Lab Fields */}
              <div className="space-y-2 max-h-[320px] overflow-y-auto pr-1">
                {LAB_CATEGORIES.map(cat => {
                  const fields = filteredFields.filter(f => f.category === cat);
                  if (fields.length === 0) return null;
                  const isExpanded = expandedCategories.has(cat);
                  const filledInCat = fields.filter(f => labValues[f.itemid] != null).length;

                  return (
                    <div key={cat} className="border border-gray-200 rounded-lg overflow-hidden">
                      <button
                        onClick={() => toggleCategory(cat)}
                        className="w-full flex items-center justify-between px-4 py-2.5 bg-gray-50 hover:bg-gray-100 transition-colors text-left"
                      >
                        <span className="text-sm font-semibold text-gray-700">{LAB_CATEGORY_LABELS[cat]}</span>
                        <span className="flex items-center gap-2">
                          {filledInCat > 0 && (
                            <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full font-medium">{filledInCat}/{fields.length}</span>
                          )}
                          {isExpanded ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
                        </span>
                      </button>
                      {isExpanded && (
                        <div className="p-3 grid grid-cols-2 gap-2">
                          {fields.map(field => (
                            <div key={field.itemid} className="flex flex-col">
                              <label className="text-xs text-gray-600 mb-0.5 truncate" title={`${field.name} (${field.unit}) [${field.normalRange}]`}>
                                {field.name} {field.unit && <span className="text-gray-400">({field.unit})</span>}
                              </label>
                              <input
                                type="number" step="any"
                                value={labValues[field.itemid] ?? ''}
                                onChange={e => handleLabChange(field.itemid, e.target.value)}
                                placeholder={field.normalRange}
                                className="border border-gray-200 rounded px-2 py-1.5 text-sm focus:ring-1 focus:ring-blue-500 outline-none"
                              />
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* STEP 3: Imaging */}
          {step === 3 && (
            <div className="space-y-6 animate-in slide-in-from-right-4 fade-in duration-300">
              <h3 className="text-xl font-bold text-gray-800">Phase B: Imaging Acquisition</h3>
              <p className="text-sm text-gray-500">Upload the primary Chest X-Ray (DICOM or PNG) for deep learning analysis.</p>
              <div
                className={cn(
                  "border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center text-center transition-colors cursor-pointer",
                  imageFile ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-gray-400 hover:bg-gray-50"
                )}
                onDragOver={e => e.preventDefault()}
                onDrop={handleFileDrop}
                onClick={() => document.getElementById('file-upload')?.click()}
              >
                <input id="file-upload" type="file" className="hidden" accept="image/png, image/jpeg, .dcm" onChange={handleFileChange} />
                {imageFile ? (
                  <>
                    <CheckCircle className="w-12 h-12 text-blue-500 mb-3" />
                    <p className="text-blue-900 font-semibold">{imageFile.name}</p>
                    <p className="text-blue-600 text-sm mt-1">{(imageFile.size / 1024 / 1024).toFixed(2)} MB</p>
                  </>
                ) : (
                  <>
                    <UploadCloud className="w-12 h-12 text-gray-400 mb-3" />
                    <p className="text-gray-700 font-medium">Click to upload or drag & drop</p>
                    <p className="text-gray-500 text-sm mt-1">PNG, JPG, or DICOM up to 20MB</p>
                  </>
                )}
              </div>
            </div>
          )}

          {/* STEP 4: Review */}
          {step === 4 && (
            <div className="space-y-6 animate-in slide-in-from-right-4 fade-in duration-300">
              <h3 className="text-xl font-bold text-gray-800 text-center mb-6">Review & Confirm</h3>
              <div className="bg-gray-50 border border-gray-200 rounded-xl p-6">
                <div className="grid grid-cols-2 gap-y-4 text-sm">
                  <div className="text-gray-500">Patient:</div>
                  <div className="font-semibold text-gray-900">{formData.firstName} {formData.lastName}</div>
                  <div className="text-gray-500">MRN:</div>
                  <div className="font-semibold text-gray-900">{formData.mrn} ({formData.age}y, {formData.sex})</div>
                  <div className="text-gray-500">Lab Values:</div>
                  <div className="font-semibold text-gray-900">{filledLabCount} of {LAB_FIELDS.length} entered</div>
                  <div className="text-gray-500">ECG Rhythm:</div>
                  <div className="font-semibold text-gray-900">{formData.rhythm}</div>
                  <div className="text-gray-500">Image Attached:</div>
                  <div className="font-semibold text-gray-900">{imageFile ? imageFile.name : 'None'}</div>
                  <div className="text-gray-500">MIMIC subject_id:</div>
                  <div className="font-semibold text-gray-900">
                    {formData.mimicSubjectId.trim()
                      ? <><span className="font-mono">{formData.mimicSubjectId.trim()}</span> <span className="text-violet-700 text-xs">→ multimodal indexing</span></>
                      : <span className="text-gray-400">None — CXR-only retrieval</span>}
                  </div>
                </div>
              </div>
              <p className="text-center text-sm text-gray-500 mt-4">
                Confirming will route this case through the Symile-MIMIC multimodal inference pipeline.
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 bg-gray-50 flex justify-between items-center">
          <button
            onClick={handlePrev}
            disabled={step === 1 || isSubmitting}
            className="px-4 py-2 text-sm font-semibold text-gray-600 hover:text-gray-900 disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-1 transition-colors"
          >
            <ChevronLeft className="w-4 h-4" /> Back
          </button>
          {step < 4 ? (
            <button
              onClick={handleNext}
              className={cn(
                "px-5 py-2.5 text-white text-sm font-semibold rounded-lg flex items-center gap-2 shadow-sm transition-all",
                hasStepErrors
                  ? "bg-blue-400 ring-2 ring-red-300 cursor-not-allowed opacity-75"
                  : "bg-blue-600 hover:bg-blue-700"
              )}
            >
              Continue <ChevronRight className="w-4 h-4" />
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={isSubmitting}
              className="px-6 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-bold rounded-lg flex items-center gap-2 shadow-sm transition-colors disabled:opacity-50"
            >
              {isSubmitting ? 'Registering...' : 'Confirm & Register'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
