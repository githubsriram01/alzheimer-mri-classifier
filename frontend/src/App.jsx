import { useState, useRef } from 'react';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';

const API_URL = 'http://127.0.0.1:8000';

const SEVERITY_ORDER = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented'];
const CLASS_COLORS = {
  NonDemented: '#7ee8b0',
  VeryMildDemented: '#ffd166',
  MildDemented: '#ffa552',
  ModerateDemented: '#ff6b6b',
};

const CARE_INFO = {
  NonDemented: {
    headline: 'No signs of dementia detected in this scan',
    message:
      'The scan pattern is consistent with typical brain aging. This is a good time to build habits that support long-term brain health.',
    tips: [
      'Keep up regular physical activity (e.g. 30 min walking, most days)',
      'Stay socially and mentally active — reading, puzzles, conversation',
      'Prioritize sleep quality and manage stress',
      'Get routine health checkups, including blood pressure and cholesterol',
    ],
    doctorType: 'General Physician',
    searchQuery: 'general physician',
  },
  VeryMildDemented: {
    headline: 'Very mild cognitive changes detected',
    message:
      'Subtle changes were picked up in this scan. At this stage, an early professional evaluation gives the most options for managing symptoms and planning ahead.',
    tips: [
      'Schedule a cognitive assessment with a specialist soon',
      'Keep a simple log of memory or concentration changes to share with a doctor',
      'Maintain a consistent daily routine and sleep schedule',
      'Involve a family member or trusted person in follow-up care',
    ],
    doctorType: 'Neurologist',
    searchQuery: 'neurologist',
  },
  MildDemented: {
    headline: 'Mild dementia-related changes detected',
    message:
      'The scan shows patterns associated with mild-stage changes. Professional evaluation is recommended to confirm findings and start a care plan.',
    tips: [
      'Book an appointment with a neurologist or geriatric specialist',
      'Set up memory aids at home — labeled rooms, calendars, reminders',
      'Review home safety (lighting, trip hazards, stove/appliance safety)',
      'Start building a support network — family, caregiver groups',
    ],
    doctorType: 'Neurologist / Geriatric Psychiatrist',
    searchQuery: 'neurologist geriatric psychiatrist',
  },
  ModerateDemented: {
    headline: 'Moderate dementia-related changes detected',
    message:
      'The scan shows patterns associated with more advanced changes. Prompt specialist care and a broader support plan are recommended.',
    tips: [
      'Seek a specialist evaluation promptly — a memory clinic can help coordinate care',
      'Discuss daily supervision and safety needs with a care team',
      'Look into local caregiver support groups — caregiving is easier with support',
      'Plan for legal/financial matters early, while decisions can involve the patient',
    ],
    doctorType: 'Neurologist / Memory Clinic',
    searchQuery: 'memory clinic neurologist',
  },
};

function CustomTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0];
  return (
    <div className="chart-tooltip">
      {p.payload.name}: {p.value.toFixed(1)}%
    </div>
  );
}

function DoctorFinder({ care }) {
  const [location, setLocation] = useState('');
  const [geoStatus, setGeoStatus] = useState('idle');
  const [coords, setCoords] = useState(null);

  const useMyLocation = () => {
    if (!navigator.geolocation) {
      setGeoStatus('error');
      return;
    }
    setGeoStatus('locating');
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setCoords({ lat: pos.coords.latitude, lng: pos.coords.longitude });
        setGeoStatus('done');
      },
      () => setGeoStatus('error'),
      { timeout: 8000 }
    );
  };

  const openMapsSearch = () => {
    const query = encodeURIComponent(`${care.searchQuery} near ${location || 'me'}`);
    let url = `https://www.google.com/maps/search/?api=1&query=${query}`;
    if (coords) {
      url = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(care.searchQuery)}&center=${coords.lat},${coords.lng}`;
    }
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  return (
    <div className="doctor-finder">
      <div className="section-title">Find a specialist near you</div>
      <div className="finder-row">
        <input
          className="text-input"
          type="text"
          placeholder="Enter your city or area"
          value={location}
          onChange={(e) => setLocation(e.target.value)}
        />
        <button className="btn-ghost" onClick={useMyLocation} type="button">
          {geoStatus === 'locating' ? 'Locating…' : '📍 Use my location'}
        </button>
      </div>
      {geoStatus === 'done' && (
        <div className="finder-hint">Location detected — search will be centered on it.</div>
      )}
      {geoStatus === 'error' && (
        <div className="finder-hint finder-hint-error">
          Couldn't get your location — you can still type a city above.
        </div>
      )}
      <button
        className="btn btn-block"
        onClick={openMapsSearch}
        disabled={!location && !coords}
      >
        Search {care.doctorType} nearby
      </button>
      <div className="finder-hint">Opens Google Maps in a new tab. No data is sent to our server.</div>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const acceptFile = (chosen) => {
    if (!chosen) return;
    setFile(chosen);
    setResult(null);
    setError(null);
    setPreview(URL.createObjectURL(chosen));
  };

  const handleFileChange = (e) => acceptFile(e.target.files[0]);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    acceptFile(e.dataTransfer.files[0]);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await axios.post(`${API_URL}/predict`, formData);
      setResult(res.data);
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(
        detail
          ? detail
          : 'Could not reach the analysis server. Make sure the backend is running on port 8000.'
      );
    } finally {
      setLoading(false);
    }
  };

  const chartData = result
    ? SEVERITY_ORDER.map((name) => ({
        name,
        probability: Math.round((result.probabilities[name] ?? 0) * 1000) / 10,
      }))
    : [];

  const care = result ? CARE_INFO[result.predicted_class] : null;

  return (
    <div className="app-root">
      <header className="top-bar">
        <div className="brand">
          <span className="brand-mark" />
          <h1>Alzheimer's MRI Classifier</h1>
        </div>
        <span className="tag">Quantum-GNN · Research demo</span>
      </header>

      <div className="stats-strip">
        <div className="stat">
          <div className="stat-value">94.97%</div>
          <div className="stat-label">Test Accuracy</div>
        </div>
        <div className="stat">
          <div className="stat-value">95.30%</div>
          <div className="stat-label">ROC-AUC</div>
        </div>
        <div className="stat">
          <div className="stat-value">92.91%</div>
          <div className="stat-label">F1 Score</div>
        </div>
        <div className="stat">
          <div className="stat-value">4-class</div>
          <div className="stat-label">Classification</div>
        </div>
      </div>

      <main className="workspace">
        {/* LEFT: upload console */}
        <aside className="console-panel">
          <div className="console-panel-inner">
            <div className="section-title">Upload scan</div>
            <div
              className={`upload-zone ${dragOver ? 'dragover' : ''} ${loading ? 'scanning' : ''}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              {preview ? (
                <img src={preview} alt="upload preview" className="upload-preview-img" />
              ) : (
                <>
                  <div className="icon">⬆</div>
                  <div className="primary-text">Click or drag an MRI slice here</div>
                  <div className="secondary-text">JPG or PNG · single 2D brain MRI slice</div>
                </>
              )}
              {loading && <div className="scan-line" />}
              <input
                ref={inputRef}
                type="file"
                accept="image/png, image/jpeg"
                onChange={handleFileChange}
              />
            </div>

            {file && (
              <div className="file-meta">
                <span className="fname">{file.name}</span>
              </div>
            )}

            <button className="btn btn-block" onClick={handleAnalyze} disabled={!file || loading}>
              {loading && <span className="spinner" />}
              {loading ? 'Analyzing MRI…' : 'Analyze MRI'}
            </button>

            {error && <div className="error-box">{error}</div>}

            <div className="pipeline-note">
              <div className="section-title" style={{ marginTop: 28 }}>Pipeline</div>
              <ol className="pipeline-list">
                <li>Skull stripping (Otsu + morphology)</li>
                <li>8×8 patch graph · 64 nodes</li>
                <li>8-qubit quantum feature circuit</li>
                <li>GATConv + SAGEConv × 2</li>
                <li>MLP classifier → 4 stages</li>
              </ol>
            </div>
          </div>
        </aside>

        {/* RIGHT: results console */}
        <section className="results-panel">
          {!result && !loading && (
            <div className="empty-state">
              <div className="empty-glyph">◎</div>
              <div className="empty-title">Awaiting scan</div>
              <div className="empty-sub">Upload an MRI slice on the left to run the classifier.</div>
            </div>
          )}

          {loading && (
            <div className="empty-state">
              <div className="empty-glyph pulsing">◎</div>
              <div className="empty-title">Running quantum-GNN inference…</div>
              <div className="empty-sub">Skull stripping → patch graph → quantum circuit → GNN. Usually a few seconds.</div>
            </div>
          )}

          {result && (
            <>
              <div className="result-header">
                <span className="label">Predicted stage</span>
                <span className="confidence-line">confidence {(result.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="predicted-class" style={{ color: CLASS_COLORS[result.predicted_class] }}>
                {result.predicted_class}
              </div>

              {result.predicted_class === 'ModerateDemented' && (
                <div className="badge-caution">
                  This class had very few training examples in the source dataset (only 64 images
                  total) — treat this prediction with extra caution.
                </div>
              )}

              <div className="grid-two">
                <div className="card">
                  <div className="section-title">Probability by stage</div>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={chartData} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
                      <XAxis
                        dataKey="name"
                        tick={{ fill: '#8fa3ac', fontSize: 10 }}
                        tickFormatter={(v) => v.replace('Demented', '')}
                        axisLine={{ stroke: '#25333c' }}
                        tickLine={false}
                      />
                      <YAxis
                        unit="%"
                        tick={{ fill: '#8fa3ac', fontSize: 10 }}
                        axisLine={{ stroke: '#25333c' }}
                        tickLine={false}
                      />
                      <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                      <Bar dataKey="probability" radius={[4, 4, 0, 0]}>
                        {chartData.map((entry) => (
                          <Cell key={entry.name} fill={CLASS_COLORS[entry.name]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="card">
                  <div className="section-title">Preprocessing &amp; attention</div>
                  <div className="image-pair">
                    <figure>
                      <img
                        src={`data:image/png;base64,${result.skull_stripped_preview}`}
                        alt="skull stripped preview"
                      />
                      <figcaption>Skull-stripped</figcaption>
                    </figure>
                    <figure>
                      <img
                        src={`data:image/png;base64,${result.attention_heatmap}`}
                        alt="attention heatmap"
                      />
                      <figcaption>Patch attention</figcaption>
                    </figure>
                  </div>
                </div>
              </div>

              {care && (
                <div className="care-section">
                  <div className="section-title">Guidance &amp; care suggestions</div>
                  <div className="grid-two">
                    <div className="card">
                      <div className="care-headline">{care.headline}</div>
                      <p className="care-message">{care.message}</p>
                      <ul className="care-tips">
                        {care.tips.map((tip) => (
                          <li key={tip}>{tip}</li>
                        ))}
                      </ul>
                    </div>
                    <DoctorFinder care={care} />
                  </div>
                </div>
              )}
            </>
          )}
        </section>
      </main>

      <footer className="foot">
        Runs locally · your image never leaves your machine
      </footer>
    </div>
  );
}
