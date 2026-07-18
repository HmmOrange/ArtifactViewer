import { useEffect, useMemo, useState } from 'react'
import qaIndex from './qa-index.json'

const PAGE_SIZE = 40
const PIPELINES = ['GraphOtter', 'SpreadsheetAgent', 'ST-raptor']
const DATASETS = ['HiTab', 'MultiHiertt']
const STATUS_LABELS = { correct: 'Correct', wrong: 'Wrong', error: 'Error' }
const ARTIFACT_SECTIONS = [
  ['input', 'Input', 'Raw workbook and query'],
  ['interpreted', 'Interpreted input', 'Parsed or preprocessed table'],
  ['workflow', 'Workflow', 'Pipeline configuration and trace'],
  ['output', 'Output', 'Generated answer and evaluation report'],
]

function StatusMark({ status, count }) {
  return <span className={`status-mark ${status}`}><i />{count ?? STATUS_LABELS[status]}</span>
}

function SelectField({ label, value, onChange, children }) {
  return (
    <label className="select-field">
      <span>{label}</span>
      <select value={value} onChange={onChange}>{children}</select>
    </label>
  )
}

function formatAnswer(answer) {
  if (Array.isArray(answer)) return answer.map((item) => String(item).trim()).join(', ')
  return answer == null ? '-' : String(answer).trim()
}

function ArtifactSections({ artifacts = {} }) {
  function reveal(path) {
    fetch(`/api/reveal?path=${encodeURIComponent(path)}`, { method: 'POST' })
  }

  return (
    <section className="artifact-grid" aria-label="Artifact files">
      {ARTIFACT_SECTIONS.map(([key, title, description], index) => {
        const paths = artifacts[key] || []
        return (
          <article className="artifact-card" key={key}>
            <div className="artifact-number">0{index + 1}</div>
            <div className="artifact-copy"><span>{title}</span><p>{description}</p></div>
            <div className="artifact-links">
              {paths.length ? paths.map((path) => (
                <button type="button" onClick={() => reveal(path)} key={path} title={`Reveal ${path} in Windows Explorer`}>
                  <code>{path}</code><b aria-hidden="true">Explorer</b>
                </button>
              )) : <em>Unavailable</em>}
            </div>
          </article>
        )
      })}
    </section>
  )
}

function App() {
  const [pipeline, setPipeline] = useState('ST-raptor')
  const [dataset, setDataset] = useState('HiTab')
  const [selectedId, setSelectedId] = useState('')
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE)

  const allRecords = qaIndex.records[pipeline]?.[dataset] || []
  const records = allRecords.slice(0, visibleCount)
  const selected = records.find((record) => record.id === selectedId) || records[0]
  const totals = useMemo(() => allRecords.reduce((counts, record) => {
    counts[record.status] += 1
    return counts
  }, { correct: 0, wrong: 0, error: 0 }), [allRecords])
  const hasMore = records.length < allRecords.length

  useEffect(() => {
    setVisibleCount(PAGE_SIZE)
    setSelectedId('')
  }, [pipeline, dataset])

  function changePipeline(event) {
    setPipeline(event.target.value)
    setDataset(DATASETS[0])
  }

  function handleScroll(event) {
    const list = event.currentTarget
    if (hasMore && list.scrollHeight - list.scrollTop - list.clientHeight < 120) {
      setVisibleCount((count) => count + PAGE_SIZE)
    }
  }

  return (
    <main className="app-shell">
      <section className="filter-bar">
        <SelectField label="Pipeline" value={pipeline} onChange={changePipeline}>
          {PIPELINES.map((name) => <option key={name}>{name}</option>)}
        </SelectField>
        <SelectField label="Dataset" value={dataset} onChange={(event) => setDataset(event.target.value)}>
          {DATASETS.map((name) => <option key={name}>{name}</option>)}
        </SelectField>
      </section>

      <section className="workspace">
        <aside className="qa-panel">
          <div className="panel-title">
            <div><span>Browse</span><h2>QA pairs</h2></div>
          </div>
          <div className="list-count">
            <StatusMark status="correct" count={totals.correct} />
            <StatusMark status="wrong" count={totals.wrong} />
            <StatusMark status="error" count={totals.error} />
          </div>
          <div className="qa-list" onScroll={handleScroll}>
            {!records.length && <div className="empty-state">No QA pairs match this selection.</div>}
            {records.map((record, index) => (
              <button className={`qa-item ${selected?.id === record.id ? 'active' : ''}`} key={record.id} onClick={() => setSelectedId(record.id)}>
                <span className="qa-index">{String(index + 1).padStart(2, '0')}</span>
                <span className="qa-question">{record.question}</span>
                <i className={`result-dot ${record.status}`} aria-label={record.status} />
              </button>
            ))}
          </div>
        </aside>

        <article className={`result-panel ${selected?.status || ''}`}>
          {selected ? <>
            <div className="result-head">
              <div><span>Selected result</span><strong>#{String(records.indexOf(selected) + 1).padStart(2, '0')}</strong></div>
              <StatusMark status={selected.status} />
            </div>
            <div className="question-block"><span>Question</span><h2>{selected.question}</h2></div>
            <div className="answer-grid">
              <section><span>Golden answer</span><p>{formatAnswer(selected.gold)}</p></section>
              <section><span>Prediction</span><p>{formatAnswer(selected.prediction)}</p></section>
            </div>
            <ArtifactSections artifacts={selected.artifacts} />
            <footer><span>Source artifact</span><code>{selected.source}</code></footer>
          </> : <div className="result-empty"><span>-</span><h2>Select a QA pair</h2><p>The golden answer and prediction will appear here.</p></div>}
        </article>
      </section>
    </main>
  )
}

export default App
