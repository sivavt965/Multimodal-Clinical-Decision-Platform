import React from 'react';
import Link from 'next/link';
import { ArrowLeft, BookOpen, HeartPulse, Network, ShieldCheck, Database, Layers, ArrowRight } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* Top Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center gap-4 sticky top-0 z-10 shadow-sm">
        <Link href="/dashboard" className="p-2 hover:bg-slate-100 rounded-full transition-colors">
          <ArrowLeft className="w-5 h-5 text-slate-600" />
        </Link>
        <div className="flex items-center gap-2">
          <BookOpen className="w-6 h-6 text-slate-800" />
          <h1 className="text-xl font-bold text-slate-900">About the Platform</h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-[800px] w-full mx-auto py-12 px-6">
        
        <article className="prose prose-slate max-w-none">
          <h1 className="text-4xl font-extrabold text-slate-900 mb-6">Multimodal Clinical Decision Support</h1>
          <p className="text-lg text-slate-600 leading-relaxed mb-12">
            The Symile-MIMIC platform represents the next generation of diagnostic augmentation. By fusing structured clinical data with unstructured medical imaging, we provide clinicians with a holistic, data-driven second opinion.
          </p>

          <div className="space-y-16">
            
            {/* The Mission */}
            <section>
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-blue-100 rounded-lg text-blue-600">
                  <HeartPulse className="w-6 h-6" />
                </div>
                <h2 className="text-2xl font-bold text-slate-800 m-0">The Mission</h2>
              </div>
              <p className="text-slate-600 leading-relaxed">
                Traditional diagnostic models often operate in silos—either analyzing a Chest X-Ray in isolation or evaluating blood labs without anatomical context. Our mission is to break down these barriers. We leverage state-of-the-art multimodal AI to evaluate a patient&apos;s complete state simultaneously, reducing cognitive load on ward doctors and surfacing high-risk divergence early.
              </p>
            </section>

            {/* How It Works */}
            <section>
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-purple-100 rounded-lg text-purple-600">
                  <Network className="w-6 h-6" />
                </div>
                <h2 className="text-2xl font-bold text-slate-800 m-0">How It Works</h2>
              </div>
              <p className="text-slate-600 leading-relaxed mb-8">
                The platform operates across four critical phases, ensuring full transparency and interpretability:
              </p>
              
              {/* Architecture Diagram Placeholder */}
              <div className="bg-white border border-slate-200 rounded-2xl p-8 shadow-sm my-8">
                <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest text-center mb-8">System Architecture Flow</h3>
                
                <div className="flex flex-col md:flex-row items-center justify-between gap-4 relative">
                  <div className="hidden md:block absolute top-1/2 left-[10%] right-[10%] h-0.5 bg-slate-100 -z-10" />
                  
                  <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl flex flex-col items-center text-center w-32 bg-white shadow-sm z-10">
                    <Database className="w-8 h-8 text-blue-500 mb-2" />
                    <span className="text-xs font-bold text-slate-700">Phase A</span>
                    <span className="text-[10px] text-slate-500 uppercase">ECG & Labs</span>
                  </div>

                  <ArrowRight className="w-6 h-6 text-slate-300 hidden md:block" />

                  <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl flex flex-col items-center text-center w-32 bg-white shadow-sm z-10">
                    <Layers className="w-8 h-8 text-purple-500 mb-2" />
                    <span className="text-xs font-bold text-slate-700">Phase B</span>
                    <span className="text-[10px] text-slate-500 uppercase">Imaging AI</span>
                  </div>

                  <ArrowRight className="w-6 h-6 text-slate-300 hidden md:block" />

                  <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl flex flex-col items-center text-center w-32 bg-white shadow-sm z-10">
                    <Network className="w-8 h-8 text-emerald-500 mb-2" />
                    <span className="text-xs font-bold text-slate-700">Phase C</span>
                    <span className="text-[10px] text-slate-500 uppercase">FAISS Retrieval</span>
                  </div>
                </div>
              </div>
            </section>

            {/* Data Ethics */}
            <section>
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-emerald-100 rounded-lg text-emerald-600">
                  <ShieldCheck className="w-6 h-6" />
                </div>
                <h2 className="text-2xl font-bold text-slate-800 m-0">Data Ethics & Privacy</h2>
              </div>
              <p className="text-slate-600 leading-relaxed">
                All models are trained strictly on the open-source <strong>MIMIC-IV</strong> dataset, ensuring robust, reproducible, and ethically sourced clinical precedents. No live patient data is used in the model tuning process, and the platform adheres strictly to HIPAA-compliant data masking pipelines when rendering the workstation.
              </p>
            </section>

          </div>
        </article>

      </main>
    </div>
  );
}
