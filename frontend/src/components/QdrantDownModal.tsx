import { endpoints } from '../api/client'
import { VectorStoreDownModal } from './VectorStoreDownModal'

const START_COMMAND = 'docker compose --profile qdrant up -d'

/** Shown when Qdrant is selected but not reachable -- Qdrant is opt-in behind compose's
 * `qdrant` profile (see docker-compose.yml), so it's simply not running until that profile is
 * used at least once. */
export function QdrantDownModal({
  open,
  onClose,
  onReachable,
  onContinueAnyway,
}: {
  open: boolean
  onClose: () => void
  onReachable: () => void
  onContinueAnyway: () => void
}) {
  return (
    <VectorStoreDownModal
      open={open}
      onClose={onClose}
      onReachable={onReachable}
      onContinueAnyway={onContinueAnyway}
      storeLabel="Qdrant"
      startCommand={START_COMMAND}
      checkReachable={() => endpoints.qdrantStatus().then((s) => s.reachable)}
    />
  )
}
