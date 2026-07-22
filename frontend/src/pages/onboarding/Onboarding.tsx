import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { useNavigate } from 'react-router-dom'
import { endpoints, errorMessage } from '../../api/client'
import { OpenSearchDownModal } from '../../components/OpenSearchDownModal'
import { QdrantDownModal } from '../../components/QdrantDownModal'
import { ProgressTrack } from './ProgressTrack'
import { StepCredentials } from './StepCredentials'
import { StepFinish, type FinishState } from './StepFinish'
import { StepLLM } from './StepLLM'
import { StepOptionalProvider } from './StepOptionalProvider'
import { StepStore } from './StepStore'
import {
  DEFAULT_EMBEDDING_PROVIDER,
  DEFAULT_RERANKER_PROVIDER,
  EMBEDDING_CHOICES,
  LLM_CHOICES,
  RERANKER_CHOICES,
} from './providers'

type StepId = 'llm' | 'creds' | 'embedding' | 'reranker' | 'store' | 'finish'

const STEP_VARIANTS = {
  enter: { opacity: 0, x: 24 },
  center: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: -24 },
}

// only relevant for restart_mode=reload (uvicorn --reload in local dev): every other mode applies
// live in-process, with no worker respawn and thus nothing to poll for -- see /api/v1/settings.
async function waitForHealth() {
  for (let i = 0; i < 60; i++) {
    try {
      await endpoints.health()
      if (i > 0) return
    } catch {
      /* still restarting */
    }
    await new Promise((r) => setTimeout(r, 1000))
  }
}

// first-run setup wizard: its own minimal, full-bleed, animated stage -- deliberately distinct
// from the dense operator-console chrome used everywhere else, since this is a new user's very
// first impression, before any of that density is earned.
export function Onboarding() {
  const navigate = useNavigate()
  const [llmProvider, setLlmProvider] = useState('litellm-ollama')
  const [embeddingProvider, setEmbeddingProvider] = useState(DEFAULT_EMBEDDING_PROVIDER)
  const [rerankerProvider, setRerankerProvider] = useState(DEFAULT_RERANKER_PROVIDER)
  const [creds, setCreds] = useState<Record<string, string>>({})
  const [vectorStore, setVectorStore] = useState('chroma')
  const [step, setStep] = useState<StepId>('llm')
  const [finishState, setFinishState] = useState<FinishState>('saving')
  const [finishDetail, setFinishDetail] = useState('')
  const [compute, setCompute] = useState('')
  const [showOsModal, setShowOsModal] = useState(false)
  const [showQdrantModal, setShowQdrantModal] = useState(false)

  useEffect(() => {
    endpoints
      .onboardingStatus()
      .then((st) => {
        // already past first-run (even if a later settings edit left something incomplete) --
        // don't trap the operator in the wizard on refresh / deep link.
        if (!st.needs_wizard) {
          navigate(st.configured ? '/query' : '/settings', { replace: true })
          return
        }
        setCompute(st.compute || '')
      })
      .catch(() => {})
  }, [navigate])

  const choice = LLM_CHOICES.find((c) => c.id === llmProvider) ?? LLM_CHOICES[0]
  const needsCreds = choice.fields.length > 0
  const steps = needsCreds
    ? ['Provider', 'Credentials', 'Embeddings', 'Reranker', 'Vector store']
    : ['Provider', 'Embeddings', 'Reranker', 'Vector store']

  const stepIndex = (() => {
    if (step === 'llm') return 0
    if (step === 'creds') return 1
    if (step === 'embedding') return needsCreds ? 2 : 1
    if (step === 'reranker') return needsCreds ? 3 : 2
    if (step === 'store') return needsCreds ? 4 : 3
    return steps.length
  })()

  async function finish() {
    setShowOsModal(false)
    setShowQdrantModal(false)
    setStep('finish')
    setFinishState('saving')
    const changes = {
      llm_provider: llmProvider,
      embedding_provider: embeddingProvider,
      reranker_provider: rerankerProvider,
      vector_store: vectorStore,
      ...creds,
    }
    try {
      const res = await endpoints.saveSettings(changes)
      if (res.restarting) {
        // dev --reload only: the worker is respawning, so wait for it to answer again.
        setFinishState('restarting')
        await waitForHealth()
      } else {
        // applied live already -- just let the success glyph register before moving on.
        setFinishState('done')
        await new Promise((r) => setTimeout(r, 700))
      }
      navigate('/query')
    } catch (e) {
      setFinishState('error')
      setFinishDetail(errorMessage(e))
    }
  }

  // opensearch / qdrant are opt-in behind compose profiles, so picking one doesn't mean it's
  // actually running yet -- check before committing, so a broken retrieval step isn't the first surprise.
  async function handleFinishClick() {
    if (vectorStore === 'opensearch') {
      const reachable = await endpoints.opensearchStatus().then((s) => s.reachable).catch(() => false)
      if (!reachable) {
        setShowOsModal(true)
        return
      }
    }
    if (vectorStore === 'qdrant') {
      const reachable = await endpoints.qdrantStatus().then((s) => s.reachable).catch(() => false)
      if (!reachable) {
        setShowQdrantModal(true)
        return
      }
    }
    finish()
  }

  const computeNote = compute
    ? `Compute: ${compute.toUpperCase()}${compute === 'cpu' ? ' — rebuild with TORCH_FLAVOR=gpu for GPU' : ''}`
    : ''

  return (
    <div className="min-h-screen scope-grid flex items-center justify-center px-6 py-16">
      <div className="grain-overlay" />
      <div className="w-full max-w-[560px]">
        {step !== 'finish' && <ProgressTrack steps={steps} currentIndex={stepIndex} />}
        <AnimatePresence mode="wait">
          <motion.div
            key={step}
            variants={STEP_VARIANTS}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.28, ease: 'easeOut' }}
          >
            {step === 'llm' && (
              <StepLLM
                selected={llmProvider}
                onSelect={setLlmProvider}
                onNext={() => setStep(needsCreds ? 'creds' : 'embedding')}
                computeNote={computeNote}
              />
            )}
            {step === 'creds' && (
              <StepCredentials
                choice={choice}
                creds={creds}
                onChange={setCreds}
                onBack={() => setStep('llm')}
                onNext={() => setStep('embedding')}
              />
            )}
            {step === 'embedding' && (
              <StepOptionalProvider
                title="Embeddings"
                subtitle="How documents get turned into vectors. Skip to use a local model — no account needed."
                skipLabel="Skip (local)"
                choices={EMBEDDING_CHOICES}
                selected={embeddingProvider}
                onSelect={setEmbeddingProvider}
                providerKey="embedding_provider"
                creds={creds}
                onChangeCreds={setCreds}
                testTarget="embedding"
                onBack={() => setStep(needsCreds ? 'creds' : 'llm')}
                onNext={() => setStep('reranker')}
                onSkip={() => {
                  setEmbeddingProvider(DEFAULT_EMBEDDING_PROVIDER)
                  setStep('reranker')
                }}
              />
            )}
            {step === 'reranker' && (
              <StepOptionalProvider
                title="Reranker"
                subtitle="Optional second-pass ranking of retrieved chunks. Skip to use a local cross-encoder."
                skipLabel="Skip (local)"
                choices={RERANKER_CHOICES}
                selected={rerankerProvider}
                onSelect={setRerankerProvider}
                providerKey="reranker_provider"
                creds={creds}
                onChangeCreds={setCreds}
                testTarget="reranker"
                onBack={() => setStep('embedding')}
                onNext={() => setStep('store')}
                onSkip={() => {
                  setRerankerProvider(DEFAULT_RERANKER_PROVIDER)
                  setStep('store')
                }}
              />
            )}
            {step === 'store' && (
              <StepStore
                selected={vectorStore}
                onSelect={setVectorStore}
                onBack={() => setStep('reranker')}
                onFinish={handleFinishClick}
              />
            )}
            {step === 'finish' && <StepFinish state={finishState} detail={finishDetail} />}
          </motion.div>
        </AnimatePresence>
      </div>
      <OpenSearchDownModal
        open={showOsModal}
        onClose={() => setShowOsModal(false)}
        onReachable={finish}
        onContinueAnyway={finish}
      />
      <QdrantDownModal
        open={showQdrantModal}
        onClose={() => setShowQdrantModal(false)}
        onReachable={finish}
        onContinueAnyway={finish}
      />
    </div>
  )
}
