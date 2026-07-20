import { errorMessage } from '../api/client'
import { ingestOne, pollJob } from '../api/ingest'
import { useHistoryStore } from '../store/historyStore'
import { toast } from '../store/toastStore'

export interface IngestBatchConfig {
  files: FileList | File[]
  indexName: string
  bucket: string
  strategy: string | null
  embProvider: string | null
  embModel: string | null
  chunkParams: Record<string, number>
  enrich: boolean
  metadata: Record<string, unknown>
}

export interface IngestBatchHooks {
  showProgress: (pct: number, label: string) => void
  hideProgress: () => void
  onBatchDone: (started: number) => void
}

// Upload + ingest a batch of files against one resolved target config. One POST per file,
// strictly serial (each job polled to a terminal state before the next upload). A failing file
// toasts and the batch continues. Mirrors ui/ingest-common.js's runIngestBatch, driven off the
// zustand history store instead of localStorage helpers directly.
export async function runIngestBatch(config: IngestBatchConfig, hooks: IngestBatchHooks): Promise<void> {
  const { files, indexName, bucket, strategy, embProvider, embModel, chunkParams, enrich, metadata } = config
  const { showProgress, hideProgress, onBatchDone } = hooks
  const { addJob, updateJob, rememberBucket } = useHistoryStore.getState()

  const fileArr = [...files]
  let started = 0
  for (let i = 0; i < fileArr.length; i++) {
    const tag = fileArr.length > 1 ? `${i + 1}/${fileArr.length} — ${fileArr[i].name}` : ''
    showProgress(0, `uploading ${tag}…`)
    try {
      const res = await ingestOne({
        file: fileArr[i],
        strategy,
        indexName,
        embProvider,
        embModel,
        chunkParams,
        metadata,
        enrich,
        onProgress: (pct) => showProgress(pct, `uploading ${tag} ${pct}%`),
      })
      started++
      addJob({
        job_id: res.job_id,
        status: res.status,
        file_name: fileArr[i].name,
        strategy: strategy ?? undefined,
        bucket,
        embedding_model: `${embProvider}/${embModel}`,
      })
      showProgress(100, `ingesting ${tag || fileArr[i].name}…`)
      await pollJob(res.job_id, (status) => updateJob(res.job_id, status))
    } catch (e) {
      toast(`${fileArr[i].name}: ${errorMessage(e)}`, 'error')
    }
  }
  if (bucket && started) rememberBucket(bucket)
  if (started) toast(fileArr.length > 1 ? `${started}/${fileArr.length} ingestion jobs finished` : 'Ingestion finished', 'success')
  hideProgress()
  onBatchDone(started)
}
