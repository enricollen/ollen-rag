import { Button } from './Button'
import { JobHistory } from './JobHistory'
import { useHistoryStore } from '../store/historyStore'
import { toast } from '../store/toastStore'

// "Jobs this session" footer shared by both ingestion flows (create-new-index and
// add-to-existing-index): a heading, a clear-list action, and the live job list itself.
export function JobsSection() {
  return (
    <div>
      <div className="flex justify-between items-center mt-6 mb-3">
        <h2 className="text-base font-semibold text-ink">Jobs this session</h2>
        <Button
          variant="secondary"
          onClick={() => {
            useHistoryStore.getState().clearJobs()
            toast('Job list cleared', 'info')
          }}
        >
          Clear list
        </Button>
      </div>
      <JobHistory />
    </div>
  )
}
