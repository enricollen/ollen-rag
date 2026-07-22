import { endpoints } from '../api/client'
import { VectorStoreDownModal } from './VectorStoreDownModal'

const START_COMMAND = 'docker compose --profile opensearch up -d'

/** Shown when OpenSearch is selected but not reachable -- OpenSearch is opt-in behind compose's
 * `opensearch` profile (see docker-compose.yml), so it's simply not running until that profile is
 * used at least once. */
export function OpenSearchDownModal({
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
      storeLabel="OpenSearch"
      startCommand={START_COMMAND}
      checkReachable={() => endpoints.opensearchStatus().then((s) => s.reachable)}
    />
  )
}
