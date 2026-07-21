import { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeRaw from 'rehype-raw'
import rehypeSanitize from 'rehype-sanitize'
import remarkGfm from 'remark-gfm'
import qaIndex from './qa-index.json'
import siflexIndex from './siflex-index.json'
import tabularModelsIndex from './tabular-models-index.json'
import mismatchIndex from './mismatch-index.json'
import failureAnalysisIndex from './failure-analysis-index.json'

const PAGE_SIZE = 40
const PIPELINES = ['GraphOtter', 'SpreadsheetAgent', 'ST-raptor']
const DATASETS = ['HiTab', 'MultiHiertt']
const DATASET_LABELS = { HiTab: 'HiTab', MultiHiertt: 'MultiHiertt / MulHi' }
const MISMATCH_PIPELINES = ['GraphOtter', 'ST-raptor', 'TableAgent-SIFLEX']
const MISMATCH_DATASETS = [...DATASETS, 'SiFlex']
const FAILURE_CLASSIFICATIONS = ['misalignment', 'misinterpretation', 'failed_code']
const FAILURE_LABELS = {
  misalignment: 'Misalignment',
  misinterpretation: 'Misinterpretation',
  failed_code: 'Failed code',
}
const STATUS_LABELS = { correct: 'Correct', wrong: 'Wrong', error: 'Error' }
const PIPELINE_PAPERS = {
  GraphOtter: { href: 'https://arxiv.org/pdf/2412.01230', venue: 'arXiv 2412.01230' },
  'ST-raptor': { href: 'https://arxiv.org/pdf/2508.18190', venue: 'arXiv 2508.18190' },
  SpreadsheetAgent: { href: 'https://aclanthology.org/2026.acl-long.86.pdf', venue: 'ACL 2026' },
}
function StatusMark({ status, count, label }) {
  return (
    <div className={`status-mark ${status}`}>
      <i />
      {count === undefined ? (label ?? STATUS_LABELS[status]) : <><b>{count}</b><small>{STATUS_LABELS[status]}</small></>}
    </div>
  )
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

function summarizeError(value) {
  const text = formatAnswer(value)
  const lines = text.split('\n').map((line) => line.trim()).filter(Boolean)
  const headline = (lines[0] || 'The pipeline did not return an answer.').replace(/^ERROR:\s*/i, '')
  const cause = [...lines].reverse().find((line) => /^[\w.]+(?:Error|Exception):/.test(line))
  return { text, headline, cause: cause === lines[0] ? '' : cause }
}

function ErrorPrediction({ value, label = 'Execution failed' }) {
  const [copied, setCopied] = useState(false)
  const { text, headline, cause } = useMemo(() => summarizeError(value), [value])

  useEffect(() => setCopied(false), [text])

  async function copyDetails() {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
    } catch {
      setCopied(false)
    }
  }

  return (
    <div className="error-prediction">
      <div className="error-summary">
        <div className="error-badge"><i />{label}</div>
        <h3>{headline}</h3>
        {cause && <p>{cause}</p>}
      </div>
      <div className="error-actions">
        <details className="error-details">
          <summary>View technical details</summary>
          <pre>{text}</pre>
        </details>
        <button type="button" className="copy-error" onClick={copyDetails}>{copied ? 'Copied' : 'Copy details'}</button>
      </div>
    </div>
  )
}

function AnswerContent({ value, markdown = false }) {
  const text = formatAnswer(value)
  if (markdown) {
    return <div className="markdown-answer"><ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw, rehypeSanitize]}>{text}</ReactMarkdown></div>
  }
  const isLong = text.length > 320 || text.split('\n').length > 6
  if (!isLong) return <p className="trace-answer">{text}</p>
  return (
    <div className="long-answer">
      <p>{text.slice(0, 280).trim()}...</p>
      <details><summary>View full answer</summary><pre>{text}</pre></details>
    </div>
  )
}

function ScoreSummary({ metrics }) {
  if (!metrics) return null
  const entries = [
    ['Overall', metrics.overall_score],
    ['Factual', metrics.factual_correctness],
    ['Coverage', metrics.coverage],
    ['Structure', metrics.structure_fidelity],
    ['Grounding', metrics.grounding],
  ].filter(([, score]) => typeof score === 'number')
  if (!entries.length) return null
  return <div className="score-summary">{entries.map(([label, score]) => <span key={label}><b>{Math.round(score * 100)}</b><small>{label}</small></span>)}</div>
}

function LlmJudgeSummary({ evaluation }) {
  if (evaluation?.method !== 'llm_judge') return null
  const hasVerdict = typeof evaluation.correct === 'boolean'
  const tone = hasVerdict ? (evaluation.correct ? 'correct' : 'wrong') : 'error'
  const confidence = typeof evaluation.confidence === 'number' ? `${Math.round(evaluation.confidence * 100)}% confidence` : ''
  const reportScore = typeof evaluation.reportScore === 'number' ? `${Math.round(evaluation.reportScore * 100)}% report score` : ''
  return (
    <div className={`llm-judge-summary ${tone}`}>
      <div className="llm-judge-head">
        <div><span>{evaluation.model || evaluation.name || 'LLM judge'}</span><strong>{hasVerdict ? (evaluation.correct ? 'Answer judged correct' : 'Answer judged incorrect') : 'Judge verdict unavailable'}</strong></div>
        <i>{[confidence, reportScore].filter(Boolean).join(' · ')}</i>
      </div>
      {evaluation.reason && <p>{evaluation.reason}</p>}
      {evaluation.error && <small>{evaluation.error}</small>}
      {evaluation.disagreesWithExactMatch && <b className="judge-disagreement">Overrides the original exact-match verdict</b>}
    </div>
  )
}

function RunMetadata({ record }) {
  const meta = record.runMeta
  if (!meta) return null
  const facts = [
    ['Model', record.model],
    ['Case', record.caseId || meta.runId],
    ['Verdict', record.verdictLabel],
    ['Turns', meta.turns],
    ['Confidence', meta.confidence],
    ['Termination', meta.terminationReason],
    ['Cost', typeof meta.cost === 'number' ? `$${meta.cost.toFixed(4)}` : null],
  ].filter(([, value]) => value !== null && value !== undefined && value !== '')
  return (
    <section className="run-metadata" aria-label="Run metadata">
      {facts.map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}
    </section>
  )
}

function embeddingTableRows(table) {
  let cellOffset = 0
  return (table.rowLengths || []).map((rowLength) => {
    const row = table.cells.slice(cellOffset, cellOffset + rowLength)
    cellOffset += rowLength
    return row
  })
}

function PklEmbeddingTable({ table }) {
  const cells = table?.cells || []
  if (!cells.length || !table.columns) return null
  const rows = embeddingTableRows(table)
  return <div className="pkl-embedding-table">
    <div className="pkl-embedding-table-head"><div><span>Source table used by GraphOtter</span><strong>{table.title}</strong></div><small>{table.rows} x {table.columns}</small></div>
    <div className="pkl-embedding-table-wrap">
      <table><tbody>{rows.map((row, rowIndex) => <tr key={rowIndex} className={rowIndex < table.headerRows ? 'header-row' : ''}>
        {row.map((cell, columnIndex) => {
          const HeaderCell = rowIndex < table.headerRows || columnIndex < table.headerColumns ? 'th' : 'td'
          return <HeaderCell key={`${cell.id}-${columnIndex}`}>{cell.value === null || cell.value === undefined || cell.value === '' ? '-' : String(cell.value)}</HeaderCell>
        })}
      </tr>)}</tbody></table>
    </div>
    <small className="pkl-caption">Headers and values restored from <code>{table.source}</code>.</small>
  </div>
}

function PklCellGraph({ table }) {
  const rows = embeddingTableRows(table).slice(0, 6)
  const columnCount = Math.min(6, Math.max(0, ...rows.map((row) => row.length)))
  if (!rows.length || !columnCount) return null
  const nodeWidth = 116
  const nodeHeight = 34
  const xStep = 146
  const yStep = 62
  const padding = 22
  const width = padding * 2 + (columnCount - 1) * xStep + nodeWidth
  const height = padding * 2 + (rows.length - 1) * yStep + nodeHeight
  const nodes = []
  const edges = []

  rows.forEach((row, rowIndex) => {
    row.slice(0, columnCount).forEach((cell, columnIndex) => {
      const x = padding + columnIndex * xStep
      const y = padding + rowIndex * yStep
      nodes.push({ cell, rowIndex, columnIndex, x, y })
      if (columnIndex > 0 && row[columnIndex - 1]) {
        edges.push({ type: 'row', x1: x - xStep + nodeWidth, y1: y + nodeHeight / 2, x2: x, y2: y + nodeHeight / 2 })
      }
      if (rowIndex > 0 && rows[rowIndex - 1]?.[columnIndex]) {
        edges.push({ type: 'column', x1: x + nodeWidth / 2, y1: y - yStep + nodeHeight, x2: x + nodeWidth / 2, y2: y })
      }
    })
  })

  const shortLabel = (value) => {
    const text = String(value === null || value === undefined || value === '' ? '-' : value)
    return text.length > 18 ? `${text.slice(0, 17)}...` : text
  }

  return <div className="pkl-cell-graph">
    <div className="pkl-cell-graph-head"><div><span>Graph retrieval edges</span><strong>Cells connect through their row and column neighborhoods</strong></div><div className="pkl-edge-legend"><i className="row" />row edge<i className="column" />column edge</div></div>
    <div className="pkl-cell-graph-canvas">
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width, height }} role="img" aria-label="GraphOtter row and column cell relationships">
        {edges.map((edge, index) => <line key={index} className={`${edge.type}-edge`} x1={edge.x1} y1={edge.y1} x2={edge.x2} y2={edge.y2} />)}
        {nodes.map(({ cell, rowIndex, columnIndex, x, y }) => {
          const header = rowIndex < table.headerRows || columnIndex < table.headerColumns
          return <g key={`${cell.id}-${rowIndex}-${columnIndex}`} className={header ? 'header-node' : ''}>
            <title>{String(cell.value ?? '-')}</title>
            <rect x={x} y={y} width={nodeWidth} height={nodeHeight} rx="5" />
            <text x={x + nodeWidth / 2} y={y + nodeHeight / 2}>{shortLabel(cell.value)}</text>
          </g>
        })}
      </svg>
    </div>
    <small className="pkl-caption">Preview shows the first {rows.length} rows and {columnCount} columns. GraphOtter can expand from a matched cell across its row and column.</small>
  </div>
}

function PklEdgeTree({ nodes = [], depth = 0, root = false }) {
  if (!nodes.length || depth > 18) return null
  return <ul className={`pkl-edge-tree${root ? ' root' : ''}`}>
    {nodes.map((node, index) => {
      const position = node.position?.some((value) => value !== null && value !== undefined) ? node.position : null
      const coordinate = position ? `r${position[0]} c${position[1]} → r${position[2]} c${position[3]}` : ''
      return <li key={`${node.label}-${index}`}>
        <div className={`pkl-tree-node${node.root ? ' tree-root' : ''}`} title={coordinate || node.label}>
          <span>{node.label}</span>
          {node.meta && <small>{node.meta}</small>}
          {coordinate && <small>{coordinate}</small>}
        </div>
        <PklEdgeTree nodes={node.children} depth={depth + 1} />
      </li>
    })}
  </ul>
}

function PklTreeDiagram({ rootLabel, rootMeta, nodes = [], tone = 'index' }) {
  if (!nodes.length) return null
  return <div className={`pkl-tree-diagram ${tone}`}>
    <PklEdgeTree root nodes={[{ label: rootLabel, meta: rootMeta, root: true, children: nodes }]} />
  </div>
}

function PklTableTree({ tree }) {
  const sections = tree?.sections || []
  if (!sections.length) return null
  const visibleSections = sections.slice(0, 6)
  const remainingSections = sections.slice(6)
  const renderSection = (section, sectionIndex, open = false) => <details className="pkl-table-section" key={section.id || sectionIndex} open={open}>
    <summary><span>{section.title}</span><small>HO-Tree topology</small></summary>
    {(section.headerTree?.length > 0 || section.bodyTree?.nodes?.length > 0) && <details className="pkl-topology-details" open>
      <summary><span>HO-Tree topology</span><small>lines show saved parent → child edges</small></summary>
      <div className="pkl-topology-canvas">
        <PklTreeDiagram rootLabel="IndexTree" rootMeta={`${section.totalColumns} leaf columns`} nodes={section.headerTree} />
        {section.bodyTree?.nodes?.length > 0 && <details className="pkl-body-tree-details">
          <summary>BodyTree · {section.bodyTree.totalBranches} row branches</summary>
          <PklTreeDiagram rootLabel="BodyTree" rootMeta={`${section.bodyTree.shownBranches} of ${section.bodyTree.totalBranches} branches shown`} nodes={section.bodyTree.nodes} tone="body" />
          {section.bodyTree.truncated && <p>Preview is capped to keep large trees readable.</p>}
        </details>}
      </div>
    </details>}
  </details>
  return <div className="pkl-table-tree">
    <p className="pkl-tree-copy">Loaded as an ST-Raptor HO-Tree. The viewer reads the saved nodes with inert classes, so no pipeline code is executed.</p>
    <div className="pkl-metrics pkl-tree-metrics">
      <span><b>{tree.nodeCounts?.features ?? '-'}</b><small>feature trees</small></span>
      <span><b>{tree.nodeCounts?.index ?? '-'}</b><small>index nodes</small></span>
      <span><b>{tree.totalColumns}</b><small>columns</small></span>
      <span><b>{tree.totalRows}</b><small>rows</small></span>
    </div>
    {visibleSections.map((section, sectionIndex) => renderSection(section, sectionIndex, sectionIndex === 0))}
    {!!remainingSections.length && <details className="pkl-more-sections">
      <summary>Show {remainingSections.length} more tree branches</summary>
      <div>{remainingSections.map((section, sectionIndex) => renderSection(section, sectionIndex + visibleSections.length))}</div>
    </details>}
  </div>
}

function WorkbookRawTable({ paths = [], recordKey = '' }) {
  const candidate = paths.map((item) => typeof item === 'string' ? { path: item, available: true } : item).find((item) => item.available && /\.xls(?:x|m)$/i.test(item.path))
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const requestId = useRef(0)

  async function loadSheet(sheetIndex = 0) {
    if (!candidate) return
    const currentRequest = ++requestId.current
    setLoading(true)
    setError('')
    try {
      const params = new URLSearchParams({ path: candidate.path, sheet: String(sheetIndex) })
      const response = await fetch(`/api/workbook-render?${params}`)
      const payload = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(payload.error || 'Could not render the source workbook.')
      if (currentRequest === requestId.current) setSummary(payload)
    } catch (requestError) {
      if (currentRequest === requestId.current) setError(requestError.message)
    } finally {
      if (currentRequest === requestId.current) setLoading(false)
    }
  }

  useEffect(() => {
    setSummary(null)
    if (candidate) loadSheet(0)
  }, [recordKey, candidate?.path])

  if (!candidate) return null
  if (!summary) return <p className={`raw-table-state${error ? ' error' : ''}`}>{error || (loading ? 'Loading source table...' : '')}</p>
  const sheet = summary.sheet
  return <>
    <div className="raw-table-scroll">
      <table className="raw-workbook-table">
        <colgroup><col className="raw-row-label" />{sheet.columns.map((column) => <col key={column.label} style={{ width: `${column.width}px` }} />)}</colgroup>
        <thead><tr><th />{sheet.columns.map((column) => <th key={column.label}>{column.label}</th>)}</tr></thead>
        <tbody>{sheet.rows.map((row) => <tr key={row.number} style={row.height ? { height: `${row.height}px` } : undefined}>
          <th>{row.number}</th>
          {row.cells.map((cell) => <td key={cell.column} colSpan={cell.colSpan} rowSpan={cell.rowSpan} style={cell.style}>{cell.value === null || cell.value === undefined ? '' : String(cell.value)}</td>)}
        </tr>)}</tbody>
      </table>
    </div>
    {(summary.sheetNames.length > 1 || sheet.truncated) && <p className="raw-table-source">
      {summary.sheetNames.length > 1 && <label>Sheet <select value={summary.sheetIndex} onChange={(event) => loadSheet(Number(event.target.value))}>{summary.sheetNames.map((name, index) => <option key={`${name}-${index}`} value={index}>{name}</option>)}</select></label>}
      {sheet.truncated && <small>Preview limited to 200 rows and 60 columns.</small>}
    </p>}
  </>
}

function PklInspector({ paths = [], recordKey = '', contextPaths = [], representation = {} }) {
  const candidates = paths.map((item) => typeof item === 'string' ? { path: item, available: true } : item).filter((item) => item.available && item.path.toLowerCase().endsWith('.pkl'))
  const contextPath = contextPaths.map((item) => typeof item === 'string' ? { path: item, available: true } : item).find((item) => item.available && item.path.toLowerCase().endsWith('.json'))?.path || ''
  const [selectedPath, setSelectedPath] = useState(candidates[0]?.path || '')
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [expanded, setExpanded] = useState(true)
  const requestId = useRef(0)
  const candidateKey = candidates.map((item) => item.path).join('|')

  async function inspect(path = selectedPath) {
    if (!path) return
    const currentRequest = ++requestId.current
    setLoading(true)
    setError('')
    try {
      const params = new URLSearchParams({ path })
      if (contextPath) params.set('context', contextPath)
      if (representation.contextId) params.set('context_id', representation.contextId)
      if (representation.tableIndex !== null && representation.tableIndex !== undefined) params.set('table_index', String(representation.tableIndex))
      const response = await fetch(`/api/pkl-summary?${params}`)
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.error || 'Could not inspect this PKL.')
      if (currentRequest !== requestId.current) return
      setSummary(payload)
    } catch (inspectError) {
      if (currentRequest !== requestId.current) return
      setSummary(null)
      setError(inspectError.message)
    } finally {
      if (currentRequest === requestId.current) setLoading(false)
    }
  }

  useEffect(() => {
    const nextPath = candidates[0]?.path || ''
    requestId.current += 1
    setSelectedPath(nextPath)
    setSummary(null)
    setError('')
    setLoading(false)
    if (nextPath) inspect(nextPath)
  }, [recordKey, candidateKey])

  function showPreview() {
    setExpanded(true)
    if (!summary && !loading) inspect(selectedPath)
  }

  function selectArtifact(event) {
    const nextPath = event.target.value
    setSelectedPath(nextPath)
    setSummary(null)
    setError('')
    inspect(nextPath)
  }

  if (!candidates.length) return null
  const embeddings = summary?.embeddings
  return (
    <div className="pkl-inspector">
      <div className="pkl-inspector-head">
        <div><span>PKL artifact</span><strong>Quick structure preview</strong></div>
        <div className="pkl-inspector-actions">
          <button type="button" className="secondary" onClick={() => expanded ? setExpanded(false) : showPreview()}>{expanded ? 'Collapse' : 'Show preview'}</button>
          {expanded && <button type="button" onClick={() => inspect(selectedPath)} disabled={loading}>{loading ? 'Loading...' : 'Refresh'}</button>}
        </div>
      </div>
      {candidates.length > 1 && <select className="pkl-select" value={selectedPath} onChange={selectArtifact}>
        {candidates.map((item) => <option key={item.path} value={item.path}>{item.path}</option>)}
      </select>}
      {expanded && error && <p className="pkl-error">{error}</p>}
      {expanded && summary && <div className="pkl-summary">
        {summary.error ? <p className="pkl-error">{summary.error}</p> : <>
          <div className="pkl-summary-title"><b>{summary.title}</b><small>{(summary.fileSize / 1024).toFixed(1)} KB · {summary.safeLoad ? 'safe preview loaded' : 'static inspection only'}</small></div>
          {summary.kind === 'table_tree' && summary.tree?.sections ? <PklTableTree tree={summary.tree} /> : summary.tree && <>
            <p className="pkl-tree-copy">ST-Raptor stores this table as a hierarchy of index and body nodes. This preview reads the structure without executing the pickle.</p>
            <div className="pkl-metrics pkl-tree-metrics">
              <span><b>{summary.tree.estimatedNodes}</b><small>tree objects</small></span>
              <span><b>{summary.tree.classCount}</b><small>node classes</small></span>
              <span><b>{summary.tree.contentPreview.length}</b><small>text samples</small></span>
              <span><b>{(summary.fileSize / 1024).toFixed(1)}</b><small>KB</small></span>
            </div>
            <div className="pkl-content-preview">
              <span>Table content found in the cache</span>
              <div>{summary.tree.contentPreview.map((item) => <i key={item}>{item}</i>)}</div>
            </div>
          </>}
          {embeddings && <>
            {summary.cellCache && <p className="pkl-tree-copy">GraphOtter flattens the interpreted table into cell nodes and stores one GTE embedding vector for each node. This view joins those vectors back to the saved table headers and values.</p>}
            <div className="pkl-metrics">
              <span><b>{embeddings.shape.join(' x ')}</b><small>shape</small></span>
              <span><b>{embeddings.dtype}</b><small>dtype</small></span>
              <span><b>{summary.rowIds?.count ?? embeddings.shape[0]}</b><small>{summary.cellCache ? 'vectors' : 'rows'}</small></span>
              <span><b>{summary.labels?.distribution?.Cell ?? summary.labels?.count ?? '-'}</b><small>labels</small></span>
            </div>
            {summary.embeddingTable && <PklCellGraph table={summary.embeddingTable} />}
            {summary.embeddingTable && <PklEmbeddingTable key={`${recordKey}-${selectedPath}`} table={summary.embeddingTable} />}
            {!summary.embeddingTable && <p className="pkl-table-note">The cache has vectors but no matching source-table context was found for this case.</p>}
          </>}
          {!embeddings && summary.object && <pre className="pkl-object-preview">{JSON.stringify(summary.object, null, 2)}</pre>}
          {!summary.safeLoad && summary.static && <details className="pkl-technical"><summary>Technical serialization details</summary><div className="pkl-static-info"><p>{summary.loadNote}</p><span>Referenced classes</span><code>{summary.static.classReferences?.join(', ') || 'none found'}</code><span>Raw string preview</span><code>{summary.static.stringPreview?.join(' · ') || 'none found'}</code></div></details>}
        </>}
      </div>}
    </div>
  )
}

function YamlStructureInspector({ paths = [], recordKey = '' }) {
  const candidates = paths.map((item) => typeof item === 'string' ? { path: item, available: true } : item).filter((item) => item.available && /\.ya?ml$/i.test(item.path))
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [expanded, setExpanded] = useState(true)
  const requestId = useRef(0)
  const path = candidates[0]?.path || ''

  useEffect(() => {
    if (!path) return
    const currentRequest = ++requestId.current
    const controller = new AbortController()
    setLoading(true)
    setError('')
    setContent('')
    const encodedPath = path.split('/').map(encodeURIComponent).join('/')
    fetch(`/api/files/${encodedPath}`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error('Could not read this structure.yaml.')
        const text = await response.text()
        if (currentRequest === requestId.current) setContent(text)
      })
      .catch((requestError) => {
        if (requestError.name !== 'AbortError' && currentRequest === requestId.current) setError(requestError.message)
      })
      .finally(() => {
        if (currentRequest === requestId.current) setLoading(false)
      })
    return () => controller.abort()
  }, [recordKey, path])

  if (!candidates.length) return null
  return <div className="yaml-inspector">
    <div className="yaml-inspector-head">
      <div><span>SiFlex Z artifact</span><strong>structure.yaml</strong></div>
      <div className="yaml-inspector-actions">
        <button type="button" className="secondary" onClick={() => setExpanded((value) => !value)}>{expanded ? 'Collapse' : 'Show YAML'}</button>
      </div>
    </div>
    {expanded && <>
      {loading && <div className="yaml-state">Reading structure.yaml...</div>}
      {error && <div className="yaml-state error">{error}</div>}
      {content && <><code className="yaml-path">{path}</code><pre className="yaml-source">{content}</pre></>}
    </>}
  </div>
}

function PathList({ paths = [], onReveal, busyPath, emptyText }) {
  if (!paths.length) return <p className="path-empty">{emptyText || 'No file was recorded for this stage.'}</p>

  function renderPath(item, index) {
    const detail = typeof item === 'string' ? { path: item, available: true } : item
    if (!detail.available) return <div className="trace-path recorded" key={`${detail.path}-${index}`} title={detail.path}><code>{detail.path}</code><b>Recorded path</b></div>
    return (
      <button type="button" onClick={() => onReveal(detail.path)} key={`${detail.path}-${index}`} disabled={busyPath === detail.path} title={`Open ${detail.path} in your file manager`}>
        <code>{detail.path}</code><b>{busyPath === detail.path ? 'Opening...' : 'Reveal'}</b>
      </button>
    )
  }

  const visiblePaths = paths.slice(0, 5)
  const hiddenPaths = paths.slice(5)
  return (
    <div className="trace-paths">
      {visiblePaths.map(renderPath)}
      {!!hiddenPaths.length && <details className="path-overflow"><summary>Show {hiddenPaths.length} more artifacts</summary><div>{hiddenPaths.map((item, index) => renderPath(item, index + visiblePaths.length))}</div></details>}
    </div>
  )
}

function TraceNode({ symbol, title, tone, paths, onReveal, busyPath, children, emptyText, status, span = 1, collapsible = false, defaultOpen = false }) {
  const [hasOpened, setHasOpened] = useState(defaultOpen)
  const header = <div className="trace-node-head">
    <span className="trace-node-symbol">{symbol}</span>
    <h3>{title}</h3>
  </div>
  const content = <>
    {children && <div className="trace-node-body">{children}</div>}
    <PathList paths={paths} onReveal={onReveal} busyPath={busyPath} emptyText={emptyText} />
  </>

  if (collapsible) {
    return <details className={`trace-node collapsible ${tone} ${status || ''} span-${span}`} open={defaultOpen} onToggle={(event) => event.currentTarget.open && setHasOpened(true)}>
      <summary>{header}</summary>
      {hasOpened && <div className="trace-node-content">{content}</div>}
    </details>
  }

  return <article className={`trace-node ${tone} ${status || ''} span-${span}`}>
    <header>{header}</header>
    {content}
  </article>
}

const STRUCTURED_WORKFLOW_PIPELINES = new Set(['GraphOtter', 'SpreadsheetAgent', 'ST-raptor'])

function WorkflowTimeline({ record, onReveal, busyPath }) {
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    setSummary(null)
    setError('')
    const params = new URLSearchParams({
      pipeline: record.pipeline,
      dataset: record.dataset,
      record: record.id,
    })
    fetch(`/api/workflow-summary?${params}`, { signal: controller.signal })
      .then(async (response) => {
        const payload = await response.json().catch(() => ({}))
        if (!response.ok) throw new Error(payload.error || 'Could not read the workflow evidence.')
        setSummary(payload)
      })
      .catch((requestError) => {
        if (requestError.name !== 'AbortError') setError(requestError.message)
      })
    return () => controller.abort()
  }, [record.id, record.pipeline, record.dataset])

  if (error) return <div className="workflow-state error"><b>Workflow summary unavailable</b><p>{error}</p></div>
  if (!summary) return <div className="workflow-state loading"><i /><span>Reading operator events from the saved artifacts...</span></div>
  if (!summary.events?.length) return <div className="workflow-state"><b>No operator events recorded</b><p>{summary.note}</p></div>

  return (
    <div className="workflow-summary">
      <div className="workflow-summary-head">
        <div><span>Execution path</span><strong>What ran, what happened, and what ran next</strong></div>
      </div>
      <ol className="workflow-timeline">
        {summary.events.map((event, index) => {
          const sourceAvailable = /^(?:Artifacts|Outputs|Datasets)\//.test(event.source || '')
          return (
            <li key={`${event.operator}-${index}`} className={`workflow-event ${event.status}`}>
              <div className="workflow-step"><span>{index + 1}</span></div>
              <article>
                <header>
                  <div><h4>{event.operator}</h4><span className={`workflow-status ${event.status}`}>{event.status}</span></div>
                </header>
                <div className="workflow-result"><b>Result</b><p>{event.result}</p></div>
                {event.fallback && <div className="workflow-fallback"><b>Fallback</b><p>{event.fallback}</p>{event.fallbackResult && <small>Fallback result: {event.fallbackResult}</small>}</div>}
                {event.evidence && <details className="workflow-evidence"><summary>Why this step is shown</summary><p>{event.evidence}</p></details>}
                {sourceAvailable ? (
                  <button type="button" className="workflow-source" onClick={() => onReveal(event.source)} disabled={busyPath === event.source} title={event.source}>
                    <code>{event.source}{event.line ? `:${event.line}` : ''}</code><b>{busyPath === event.source ? 'Opening' : 'Open evidence'}</b>
                  </button>
                ) : (
                  <div className="workflow-source documented" title={event.source}><code>{event.source}{event.line ? `:${event.line}` : ''}</code><b>Source code</b></div>
                )}
              </article>
            </li>
          )
        })}
      </ol>
      <p className="workflow-note">{summary.note}</p>
    </div>
  )
}

function TraceFlow({ record }) {
  const [busyPath, setBusyPath] = useState('')
  const [notice, setNotice] = useState(null)
  const artifacts = record.artifacts || {}
  const inputPaths = artifacts.input || []
  const queryPaths = inputPaths.filter((path) => /\.(?:jsonl?|csv)$/i.test(path))
  const rawPaths = inputPaths.filter((path) => !queryPaths.includes(path))
  const components = record.components || {}
  const componentPaths = (key, fallback) => components[key]?.paths || fallback
  const renderMarkdownAnswers = ['TableAgent-SIFLEX', 'Tabular-Models'].includes(record.pipeline)
  const interpretedPaths = componentPaths('Z', artifacts.interpreted || [])

  useEffect(() => {
    if (!notice) return undefined
    const timeout = window.setTimeout(() => setNotice(null), 3500)
    return () => window.clearTimeout(timeout)
  }, [notice])

  async function reveal(path) {
    setBusyPath(path)
    setNotice(null)
    try {
      const response = await fetch(`/api/reveal?path=${encodeURIComponent(path)}`, { method: 'POST' })
      const payload = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(payload.error || 'Could not open this artifact.')
      setNotice({ type: 'success', text: 'Opened in your file manager.' })
    } catch (error) {
      setNotice({ type: 'error', text: error.message })
    } finally {
      setBusyPath('')
    }
  }

  return (
    <>
      <RunMetadata record={record} />
      {notice && <div className={`artifact-toast ${notice.type}`} role="status" aria-live="polite">{notice.text}</div>}
      <section className="trace-grid" aria-label="Pipeline components">
        <TraceNode symbol="q" title="Query" tone="query" paths={componentPaths('q', queryPaths)} onReveal={reveal} busyPath={busyPath} emptyText="The question is stored in the indexed report." span={3} collapsible defaultOpen>
          <p className="trace-question">{record.question}</p>
        </TraceNode>
        <TraceNode symbol="X" title="Raw data" tone="raw" paths={componentPaths('X', rawPaths.length ? rawPaths : inputPaths)} onReveal={reveal} busyPath={busyPath} span={3} collapsible>
          <WorkbookRawTable paths={componentPaths('X', rawPaths.length ? rawPaths : inputPaths)} recordKey={record.id} />
        </TraceNode>
        <TraceNode symbol="Z" title="Interpreted representation" tone="interpreted" paths={interpretedPaths} onReveal={reveal} busyPath={busyPath} span={3} collapsible>
          <PklInspector paths={interpretedPaths} contextPaths={[...interpretedPaths, ...inputPaths]} representation={record.representation || {}} recordKey={record.id} />
          <YamlStructureInspector paths={interpretedPaths} recordKey={record.id} />
        </TraceNode>
        <TraceNode symbol="W" title="Solving workflow" tone="workflow" paths={componentPaths('W', artifacts.workflow || [])} onReveal={reveal} busyPath={busyPath} span={3} collapsible>
          {STRUCTURED_WORKFLOW_PIPELINES.has(record.pipeline) && <WorkflowTimeline record={record} onReveal={reveal} busyPath={busyPath} />}
        </TraceNode>
        <TraceNode symbol="Y" title="Pipeline result" tone="result" status={record.status} paths={componentPaths('Y', artifacts.output || [])} onReveal={reveal} busyPath={busyPath}>
          {record.status === 'error' ? <ErrorPrediction value={record.prediction} label={record.verdictLabel === 'Insufficient' ? 'Insufficient answer' : undefined} /> : <AnswerContent value={record.prediction} markdown={renderMarkdownAnswers} />}
          <LlmJudgeSummary evaluation={record.evaluation} />
          <ScoreSummary metrics={record.metrics} />
        </TraceNode>
        <TraceNode symbol="Y*" title="Reference answer" tone="reference" paths={componentPaths('Y*', queryPaths)} onReveal={reveal} busyPath={busyPath} emptyText="Reference answer path is included with the query source.">
          {record.gold == null && record.referenceNote ? <div className="reference-unavailable"><b>Reference not bundled</b><p>{record.referenceNote}</p><StatusMark status={record.status} label={record.verdictLabel} /></div> : <AnswerContent value={record.gold} markdown={renderMarkdownAnswers} />}
        </TraceNode>
      </section>
    </>
  )
}

function failurePercent(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`
}

function FailureRatioBar({ label, count, denominator, ratio, tone }) {
  return <div className={`failure-ratio-row ${tone}`}>
    <div><span>{label}</span><strong>{count}<small>/{denominator}</small></strong></div>
    <div className="failure-ratio-track"><i style={{ width: `${Math.max(0, Number(ratio || 0) * 100)}%` }} /></div>
    <b>{failurePercent(ratio)}</b>
  </div>
}

function FailureSummaryCard({ item }) {
  return <article className="failure-summary-card">
    <header><div><span>{item.benchmark}</span><strong>{item.solution}</strong></div><small>{item.totalFailCases} failed outcomes</small></header>
    <FailureRatioBar label="Misalignment" count={item.misalignmentCases} denominator={item.totalWrongCases} ratio={item.misalignmentRatio} tone="misalignment" />
    <FailureRatioBar label="Misinterpretation" count={item.misinterpretationCases} denominator={item.totalWrongCases} ratio={item.misinterpretationRatio} tone="misinterpretation" />
    <FailureRatioBar label="Failed code" count={item.failedCodeCases} denominator={item.totalFailCases} ratio={item.failedCodeRatio} tone="failed-code" />
  </article>
}

function parsedPreview(value) {
  if (!value) return null
  try {
    return JSON.parse(value)
  } catch {
    return null
  }
}

function FailureMatrix({ matrix }) {
  if (!Array.isArray(matrix) || !matrix.length) return <div className="failure-preview-empty">No matrix preview was saved.</div>
  return <div className="failure-matrix-scroll"><table><tbody>
    {matrix.map((row, rowIndex) => <tr key={rowIndex}>{(Array.isArray(row) ? row : [row]).map((cell, columnIndex) => {
      const Cell = rowIndex === 0 || columnIndex === 0 ? 'th' : 'td'
      return <Cell key={columnIndex}>{cell === '' || cell == null ? <i>-</i> : String(cell)}</Cell>
    })}</tr>)}
  </tbody></table></div>
}

function FailurePreview({ value, label }) {
  const parsed = parsedPreview(value)
  if (Array.isArray(parsed)) return <FailureMatrix matrix={parsed} />
  if (parsed && typeof parsed === 'object') {
    return <div className="failure-table-previews">{Object.entries(parsed).map(([table, matrix]) => <section key={table}><span>{label} table {table}</span><FailureMatrix matrix={matrix} /></section>)}</div>
  }
  if (!value) return <div className="failure-preview-empty">No preview was saved.</div>
  return <pre className="failure-text-preview">{value}</pre>
}

function FailureArtifactPaths({ symbol, paths = [], onReveal, busyPath }) {
  if (!paths.length) return <div className="failure-path-empty">No saved {symbol} artifact.</div>
  return <div className="failure-artifact-paths">{paths.map((path) => <button type="button" key={path} onClick={() => onReveal(path)} disabled={busyPath === path} title={path}><code>{path}</code><b>{busyPath === path ? 'Opening' : 'Open'}</b></button>)}</div>
}

function FailureNativeZVisual({ record }) {
  const zPaths = record.paths?.Z || []
  if (!zPaths.some((path) => /\.pkl$/i.test(path))) return null
  const isGraphOtter = record.solution === 'GraphOtter'
  const isMulhi = record.benchmark === 'mulhi'
  const selectedIndex = Number(record.selectedTables?.[0])
  const representation = {
    contextId: isGraphOtter ? (isMulhi ? record.sampleId : String(record.expectedTables?.[0] || '')) : '',
    tableIndex: isGraphOtter && isMulhi && Number.isInteger(selectedIndex) ? selectedIndex : null,
  }
  return <div className="failure-native-z-visual">
    <div className="failure-native-z-label"><span>Native pipeline visual</span><strong>{isGraphOtter ? 'GraphOtter cell graph and embedding table' : 'ST-Raptor HO-Tree'}</strong></div>
    <PklInspector paths={zPaths} contextPaths={[...zPaths, ...(record.paths?.X || [])]} representation={representation} recordKey={record.id} />
  </div>
}

function FailureCaseDetail({ record }) {
  const [busyPath, setBusyPath] = useState('')
  const [notice, setNotice] = useState(null)

  useEffect(() => {
    if (!notice) return undefined
    const timeout = window.setTimeout(() => setNotice(null), 3500)
    return () => window.clearTimeout(timeout)
  }, [notice])

  async function reveal(path) {
    setBusyPath(path)
    setNotice(null)
    try {
      const response = await fetch(`/api/reveal?path=${encodeURIComponent(path)}`, { method: 'POST' })
      const payload = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(payload.error || 'Could not open this artifact.')
      setNotice({ type: 'success', text: 'Opened in your file manager.' })
    } catch (error) {
      setNotice({ type: 'error', text: error.message })
    } finally {
      setBusyPath('')
    }
  }

  const gold = formatAnswer(record.gold)
  const isCodeFailure = record.outcome === 'execution_failed'
  const xPaths = record.paths?.X || []
  const hasWorkbook = xPaths.some((path) => /\.xls(?:x|m)$/i.test(path))
  return <div className="failure-case-detail">
    {notice && <div className={`artifact-toast ${notice.type}`} role="status">{notice.text}</div>}
    <header className="failure-case-head">
      <div><span>{record.benchmark} / {record.solution}</span><h2>{FAILURE_LABELS[record.classification]}</h2></div>
      <div className={`failure-classification ${record.classification}`}><i />{record.classification.replace('_', ' ')}</div>
    </header>
    <div className="failure-case-meta"><span>Case <code>{record.sampleId}</code></span><span>W <b>{record.wStatus}</b></span>{!isCodeFailure && <span>Z <b>{record.zMatchesX ? 'matches X' : 'does not match X'}</b></span>}</div>
    <section className="failure-question"><span>Question</span><p>{record.question}</p></section>
    {!isCodeFailure && <section className={`failure-verification ${record.zMatchesX ? 'matched' : 'mismatched'}`}>
      <div><span>X to Z verification</span><strong>{record.zMatchesX ? 'Representation aligned' : 'Representation mismatch'}</strong></div>
      <p>{record.xZEvidence}</p>
      <div className="failure-checks"><span className={record.zCreated ? 'pass' : 'fail'}>Z created</span><span className={record.zStructuralMatch ? 'pass' : 'fail'}>Structure</span><span className={record.zScopeMatch ? 'pass' : 'fail'}>Relevant scope</span></div>
    </section>}
    <div className="failure-trace-grid">
      <article className="failure-trace-node raw">
        <header><b>X</b><div><span>Expected raw evidence</span><strong>{JSON.stringify(record.expectedTables)}</strong></div></header>
        {hasWorkbook ? <div className="failure-native-x-visual"><WorkbookRawTable paths={xPaths} recordKey={record.id} /></div> : <FailurePreview value={record.xPreview} label="X" />}
        <FailureArtifactPaths symbol="X" paths={xPaths} onReveal={reveal} busyPath={busyPath} />
      </article>
      <article className="failure-trace-node interpreted">
        <header><b>Z</b><div><span>Selected representation</span><strong>{JSON.stringify(record.selectedTables)}</strong></div></header>
        <FailureNativeZVisual record={record} />
        <FailurePreview value={record.zPreview} label="Z" />
        <FailureArtifactPaths symbol="Z" paths={record.paths?.Z} onReveal={reveal} busyPath={busyPath} />
      </article>
      <article className={`failure-trace-node workflow ${record.outcome === 'execution_failed' ? 'failed' : ''}`}>
        <header><b>W</b><div><span>Solving workflow</span><strong>{record.wStatus}</strong></div></header>
        {record.wError ? <ErrorPrediction value={record.wError} /> : <div className="failure-workflow-complete"><i />Execution completed and returned an answer.</div>}
        <FailureArtifactPaths symbol="W" paths={record.paths?.W} onReveal={reveal} busyPath={busyPath} />
      </article>
      <article className="failure-trace-node answers">
        <header><b>Y</b><div><span>Answer comparison</span><strong>Prediction vs reference</strong></div></header>
        <div className="failure-answer-pair"><section><span>Y</span><p>{record.prediction || '-'}</p></section><section><span>Y*</span><p>{gold}</p></section></div>
        {record.judgeReason && <div className="failure-judge-reason"><span>LLM judge</span><p>{record.judgeReason}</p></div>}
        <FailureArtifactPaths symbol="Y" paths={record.paths?.Y} onReveal={reveal} busyPath={busyPath} />
      </article>
    </div>
    {record.goldTableEvidence?.length > 0 && <footer className="failure-gold-evidence"><span>Gold table evidence</span><code>{JSON.stringify(record.goldTableEvidence)}</code></footer>}
  </div>
}

function downloadFailureCsv(records) {
  const columns = ['benchmark', 'solution', 'sampleId', 'classification', 'outcome', 'question', 'prediction', 'gold', 'wStatus', 'zMatchesX', 'expectedTables', 'selectedTables', 'xZEvidence']
  const quote = (value) => `"${String(typeof value === 'string' ? value : JSON.stringify(value ?? '')).replaceAll('"', '""')}"`
  const csv = `\ufeff${columns.map(quote).join(',')}\r\n${records.map((record) => columns.map((column) => quote(record[column])).join(',')).join('\r\n')}`
  const link = document.createElement('a')
  link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
  link.download = 'filtered_failure_cases.csv'
  link.click()
  URL.revokeObjectURL(link.href)
}

function FailureAnalysisView() {
  const [benchmark, setBenchmark] = useState('all')
  const [solution, setSolution] = useState('all')
  const [classification, setClassification] = useState('all')
  const [query, setQuery] = useState('')
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE)
  const [selectedId, setSelectedId] = useState('')
  const records = failureAnalysisIndex.records || []
  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    return records.filter((record) => (
      (benchmark === 'all' || record.benchmark === benchmark)
      && (solution === 'all' || record.solution === solution)
      && (classification === 'all' || record.classification === classification)
      && (!normalized || `${record.question} ${record.sampleId} ${record.prediction} ${formatAnswer(record.gold)}`.toLowerCase().includes(normalized))
    ))
  }, [records, benchmark, solution, classification, query])
  const visible = filtered.slice(0, visibleCount)
  const selected = visible.find((record) => record.id === selectedId) || visible[0]
  const categoryCounts = useMemo(() => filtered.reduce((counts, record) => {
    counts[record.classification] += 1
    return counts
  }, { misalignment: 0, misinterpretation: 0, failed_code: 0 }), [filtered])

  useEffect(() => {
    setVisibleCount(PAGE_SIZE)
    setSelectedId('')
  }, [benchmark, solution, classification, query])

  function handleScroll(event) {
    const list = event.currentTarget
    if (visible.length < filtered.length && list.scrollHeight - list.scrollTop - list.clientHeight < 120) {
      setVisibleCount((count) => count + PAGE_SIZE)
    }
  }

  if (!failureAnalysisIndex.available) return <section className="failure-analysis-unavailable"><h2>Failure analysis is not available</h2><p>{failureAnalysisIndex.note}</p></section>
  return <section className="failure-analysis-view">
    <div className="failure-analysis-scroll">
      <div className="failure-summary-grid">{failureAnalysisIndex.summary.map((item) => <FailureSummaryCard key={`${item.benchmark}-${item.solution}`} item={item} />)}</div>
      <section className="failure-explorer">
        <div className="failure-filter-bar">
          <div className="failure-filter-copy"><span>Case explorer</span><strong>{filtered.length} matching failed cases</strong><small>{categoryCounts.misalignment} misalignment · {categoryCounts.misinterpretation} misinterpretation · {categoryCounts.failed_code} failed code</small></div>
          <SelectField label="Benchmark" value={benchmark} onChange={(event) => setBenchmark(event.target.value)}><option value="all">All benchmarks</option><option>Hitab</option><option>mulhi</option></SelectField>
          <SelectField label="Solution" value={solution} onChange={(event) => setSolution(event.target.value)}><option value="all">All solutions</option>{PIPELINES.map((name) => <option key={name}>{name}</option>)}</SelectField>
          <SelectField label="Failure mode" value={classification} onChange={(event) => setClassification(event.target.value)}><option value="all">All failure modes</option>{FAILURE_CLASSIFICATIONS.map((name) => <option key={name} value={name}>{FAILURE_LABELS[name]}</option>)}</SelectField>
          <label className="search-field"><span>Find a case</span><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Question, case ID, or answer..." /></label>
          <button type="button" className="failure-export" onClick={() => downloadFailureCsv(filtered)}>Export filtered CSV</button>
        </div>
        <div className="failure-workspace">
          <aside className="failure-case-list" onScroll={handleScroll} tabIndex={0}>
            {!visible.length && <div className="empty-state">No failed cases match these filters.</div>}
            {visible.map((record, index) => <button type="button" key={record.id} className={`failure-case-item ${record.classification} ${selected?.id === record.id ? 'active' : ''}`} onClick={() => setSelectedId(record.id)}>
              <span>{String(index + 1).padStart(3, '0')}</span><div><b>{record.question}</b><small>{record.benchmark} · {record.solution} · {FAILURE_LABELS[record.classification]}</small></div><i />
            </button>)}
          </aside>
          <article className="failure-detail-panel">{selected ? <FailureCaseDetail record={selected} /> : <div className="result-empty"><span>-</span><h2>Select a failed case</h2></div>}</article>
        </div>
      </section>
    </div>
  </section>
}

function App() {
  const [viewMode, setViewMode] = useState('benchmarks')
  const [pipeline, setPipeline] = useState('ST-raptor')
  const [dataset, setDataset] = useState('HiTab')
  const [selectedId, setSelectedId] = useState('')
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE)
  const [query, setQuery] = useState('')
  const [tabularModel, setTabularModel] = useState(tabularModelsIndex.models[0]?.id || '')
  const [tabularVerdict, setTabularVerdict] = useState('all')
  const [mismatchPipeline, setMismatchPipeline] = useState('all')
  const [mismatchDataset, setMismatchDataset] = useState('all')

  const mismatchRecords = useMemo(() => {
    const validations = new Map(mismatchIndex.records.map((item) => [item.id, item.validation]))
    const benchmarkRecords = PIPELINES.flatMap((pipelineName) => (
      DATASETS.flatMap((datasetName) => qaIndex.records[pipelineName]?.[datasetName] || [])
    ))
    return [...benchmarkRecords, ...siflexIndex.records].filter((record) => validations.has(record.id)).map((record) => ({
      ...record,
      representationValidation: validations.get(record.id),
    }))
  }, [])

  const allRecords = useMemo(() => {
    if (viewMode === 'siflex') return siflexIndex.records
    if (viewMode === 'mismatches') {
      return mismatchRecords.filter((record) => (
        (mismatchPipeline === 'all' || record.pipeline === mismatchPipeline)
        && (mismatchDataset === 'all' || record.dataset === mismatchDataset)
      ))
    }
    if (viewMode === 'tabular') {
      return tabularModelsIndex.records.filter((record) => (
        (!tabularModel || record.modelId === tabularModel)
        && (tabularVerdict === 'all' || record.category === tabularVerdict)
      ))
    }
    return qaIndex.records[pipeline]?.[dataset] || []
  }, [viewMode, pipeline, dataset, tabularModel, tabularVerdict, mismatchRecords, mismatchPipeline, mismatchDataset])
  const filteredRecords = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase()
    if (!normalizedQuery) return allRecords
    return allRecords.filter((record) => String(record.question).toLowerCase().includes(normalizedQuery))
  }, [allRecords, query])
  const records = filteredRecords.slice(0, visibleCount)
  const selected = records.find((record) => record.id === selectedId) || records[0]
  const totals = useMemo(() => allRecords.reduce((counts, record) => {
    counts[record.status] += 1
    return counts
  }, { correct: 0, wrong: 0, error: 0 }), [allRecords])
  const hasMore = records.length < filteredRecords.length

  useEffect(() => {
    setVisibleCount(PAGE_SIZE)
    setSelectedId('')
  }, [viewMode, pipeline, dataset, tabularModel, tabularVerdict, mismatchPipeline, mismatchDataset, query])

  function changePipeline(event) {
    setPipeline(event.target.value)
    setDataset(DATASETS[0])
  }

  function changeView(mode) {
    setViewMode(mode)
    setQuery('')
    if (mode === 'tabular') setTabularVerdict('all')
  }

  const pipelinePaper = PIPELINE_PAPERS[pipeline]

  function handleScroll(event) {
    const list = event.currentTarget
    if (hasMore && list.scrollHeight - list.scrollTop - list.clientHeight < 120) {
      setVisibleCount((count) => count + PAGE_SIZE)
    }
  }

  return (
    <main className={`app-shell ${viewMode === 'analysis' ? 'analysis-shell' : ''}`}>
      <nav className="view-tabs" aria-label="Viewer sections">
        <button type="button" className={viewMode === 'benchmarks' ? 'active' : ''} onClick={() => changeView('benchmarks')}><b>Benchmark pipelines</b><span>GraphOtter, SpreadsheetAgent, ST-Raptor</span></button>
        <button type="button" className={viewMode === 'analysis' ? 'active' : ''} onClick={() => changeView('analysis')}><b>Failure analysis</b><span>Ratios and all {failureAnalysisIndex.totalCases || 0} failed cases</span></button>
        <button type="button" className={viewMode === 'mismatches' ? 'active' : ''} onClick={() => changeView('mismatches')}><b>Answer mismatches</b><span>Z validated against X, W completed, Y differs from Y*</span></button>
        <button type="button" className={viewMode === 'siflex' ? 'active' : ''} onClick={() => changeView('siflex')}><b>TableAgent / SIFLEX</b><span>{siflexIndex.run.name}</span></button>
        <button type="button" className={viewMode === 'tabular' ? 'active' : ''} onClick={() => changeView('tabular')}><b>Tabular models</b><span>Runs matched to uploaded Data Lake files</span></button>
      </nav>

      {viewMode === 'analysis' ? <FailureAnalysisView /> : <>
      <section className={`filter-bar ${viewMode === 'siflex' ? 'siflex-filter' : ''} ${viewMode === 'tabular' ? 'tabular-filter' : ''} ${viewMode === 'mismatches' ? 'mismatch-filter' : ''}`} aria-label="Filters">
        {viewMode === 'benchmarks' ? <>
          <div className="filter-copy pipeline-reference">
            <span>Pipeline reference</span>
            <strong>{pipeline}</strong>
            <a href={pipelinePaper.href} target="_blank" rel="noreferrer" title={`Open the ${pipeline} paper`}>
              <i />Read paper <small>{pipelinePaper.venue}</small><b aria-hidden="true" />
            </a>
          </div>
          <SelectField label="Pipeline" value={pipeline} onChange={changePipeline}>
            {PIPELINES.map((name) => <option key={name}>{name}</option>)}
          </SelectField>
          <SelectField label="Dataset" value={dataset} onChange={(event) => setDataset(event.target.value)}>
            {DATASETS.map((name) => <option key={name} value={name}>{DATASET_LABELS[name]}</option>)}
          </SelectField>
        </> : viewMode === 'mismatches' ? <>
          <div className="filter-copy mismatch-filter-copy"><span>Validated mismatch cases</span><strong>{allRecords.length} runs with Z matching X and W completed</strong></div>
          <SelectField label="Pipeline" value={mismatchPipeline} onChange={(event) => setMismatchPipeline(event.target.value)}>
            <option value="all">All pipelines</option>
            {MISMATCH_PIPELINES.map((name) => <option key={name} value={name}>{name}</option>)}
          </SelectField>
          <SelectField label="Dataset" value={mismatchDataset} onChange={(event) => setMismatchDataset(event.target.value)}>
            <option value="all">All datasets</option>
            {MISMATCH_DATASETS.map((name) => <option key={name} value={name}>{DATASET_LABELS[name] || name}</option>)}
          </SelectField>
        </> : viewMode === 'siflex' ? <>
          <div className="filter-copy"><span>SIFLEX run</span><strong>{siflexIndex.run.name}</strong></div>
          <div className="run-summary"><span><b>{siflexIndex.run.pass}</b> passed</span><span><b>{siflexIndex.run.fail}</b> failed</span><span><b>{Math.round((siflexIndex.run.pass_rate || 0) * 100)}%</b> pass rate</span></div>
        </> : <>
          <div className="filter-copy"><span>Uploaded Data Lake subset</span><strong>{tabularModelsIndex.root.matchedFiles} source files · {tabularModelsIndex.root.records} matched runs</strong></div>
          <SelectField label="Model" value={tabularModel} onChange={(event) => setTabularModel(event.target.value)}>
            {tabularModelsIndex.models.map((model) => <option key={model.id} value={model.id}>{model.label} ({model.records})</option>)}
          </SelectField>
          <SelectField label="Evaluator verdict" value={tabularVerdict} onChange={(event) => setTabularVerdict(event.target.value)}>
            <option value="all">All verdicts</option>
            <option value="Correct">Correct</option>
            <option value="Incorrect">Incorrect</option>
            <option value="Insufficient">Insufficient</option>
          </SelectField>
        </>}
        <label className="search-field">
          <span>Find a question</span>
          <input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search the question text..." />
        </label>
      </section>

      <section className="workspace">
        <aside className="qa-panel">
          <div className="panel-title">
            <div><span>Browse the run</span><h2>{viewMode === 'siflex' ? 'SIFLEX cases' : viewMode === 'tabular' ? 'Model cases' : viewMode === 'mismatches' ? 'Answer mismatches' : 'QA pairs'}</h2><p>{viewMode === 'mismatches' ? 'Every Z is validated against X before the answer mismatch is included.' : 'Select a question to open its full trace.'}</p></div>
          </div>
          {viewMode === 'mismatches' ? <div className="mismatch-list-criteria"><span><b>Z</b> matches X</span><span><b>W</b> completed</span><span className="different"><b>Y</b> != Y*</span></div> : <div className="list-count"><StatusMark status="correct" count={totals.correct} /><StatusMark status="wrong" count={totals.wrong} /><StatusMark status="error" count={totals.error} /></div>}
          <div className="qa-list" onScroll={handleScroll} tabIndex={0} aria-label="Question results">
            {!records.length && <div className="empty-state">No QA pairs match this search.</div>}
            {records.map((record, index) => (
              <button className={`qa-item ${selected?.id === record.id ? 'active' : ''}`} key={record.id} onClick={() => setSelectedId(record.id)}>
                <span className="qa-index">{String(index + 1).padStart(2, '0')}</span>
                <span className="qa-item-copy"><span className="qa-question">{record.question}</span>{viewMode === 'mismatches' && <small>{record.pipeline} / {DATASET_LABELS[record.dataset] || record.dataset}</small>}</span>
                <i className={`result-dot ${record.status}`} aria-label={record.status} />
              </button>
            ))}
          </div>
        </aside>

        <article className={`result-panel ${selected?.status || ''}`} tabIndex={0} aria-label="Selected result details">
          {selected ? <>
            <div className="result-head">
              <div><span>Selected result</span><strong>#{String(records.indexOf(selected) + 1).padStart(2, '0')}</strong></div>
              <StatusMark status={selected.status} label={selected.verdictLabel} />
            </div>
            <TraceFlow record={selected} />
          </> : <div className="result-empty"><span>-</span><h2>Select a QA pair</h2><p>The golden answer and prediction will appear here.</p></div>}
        </article>
      </section>
      </>}
    </main>
  )
}

export default App
