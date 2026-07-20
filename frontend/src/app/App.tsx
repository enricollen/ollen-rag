import { Navigate, HashRouter, Route, Routes } from 'react-router-dom'
import { ToastHost } from '../components/ToastHost'
import { Onboarding } from '../pages/onboarding/Onboarding'
import { SettingsPage } from '../pages/settings/SettingsPage'
import { IndicesPage } from '../pages/indices/IndicesPage'
import { IngestionPage } from '../pages/ingestion/IngestionPage'
import { RetrievalPage } from '../pages/retrieval/RetrievalPage'
import { QueryPage } from '../pages/query/QueryPage'
import { EvalPage } from '../pages/eval/EvalPage'
import { Shell } from './Shell'

export function App() {
  return (
    <HashRouter>
      <ToastHost />
      <Routes>
        <Route path="/welcome" element={<Onboarding />} />
        <Route path="/" element={<Shell />}>
          <Route index element={<Navigate to="/ingestion" replace />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="indices" element={<IndicesPage />} />
          <Route path="ingestion" element={<IngestionPage />} />
          <Route path="retrieval" element={<RetrievalPage />} />
          <Route path="query" element={<QueryPage />} />
          <Route path="eval" element={<EvalPage />} />
          <Route path="*" element={<Navigate to="/ingestion" replace />} />
        </Route>
      </Routes>
    </HashRouter>
  )
}
