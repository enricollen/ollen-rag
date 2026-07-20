import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { useNavigate } from 'react-router-dom'
import { endpoints, errorMessage } from '../../api/client'
import { OpenSearchDownModal } from '../../components/OpenSearchDownModal'
import { ProgressTrack } from './ProgressTrack'
import { StepCredentials } from './StepCredentials'
import { StepFinish, type FinishState } from './StepFinish'
import { StepLLM } from './StepLLM'
import { StepStore } from './StepStore'
import { LLM_CHOICES } from './providers'

type StepId = 'llm' | 'creds' | 'store' | 'finish'

const STEP_VARIANTS = {
  enter: { opacity: 0, x: 24 },
  center: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: -24 },
}

// Only relevant for restart_mode=reload (uvicorn --reload in local dev): every other mode applies
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

// First-run setup wizard: its own minimal, full-bleed, animated stage -- deliberately distinct
// from the dense operator-console chrome used everywhere else, since this is a new user's very
// first impression, before any of that density is earned.
export function Onboarding() {
  const navigate = useNavigate()
  const [llmProvider, setLlmProvider] = useState('litellm-ollama')
  const [creds, setCreds] = useState<Record<string, string>>({})
  const [vectorStore, setVectorStore] = useState('chroma')
  const [step, setStep] = useState<StepId>('llm')
  const [finishState, setFinishState] = useState<FinishState>('saving')
  const [finishDetail, setFinishDetail] = useState('')
  const [compute, setCompute] = useState('')
  const [showOsModal, setShowOsModal] = useState(false)

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
  const steps = needsCreds ? ['Provider', 'Credentials', 'Vector store'] : ['Provider', 'Vector store']
  const stepIndex = step === 'llm' ? 0 : step === 'creds' ? 1 : step === 'store' ? (needsCreds ? 2 : 1) : steps.length

  async function finish() {
    setShowOsModal(false)
    setStep('finish')
    setFinishState('saving')
    const changes = {
      llm_provider: llmProvider,
      embedding_provider: 'fastembed',
      reranker_provider: 'sentence-transformers',
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
        // Applied live already -- just let the success glyph register before moving on.
        setFinishState('done')
        await new Promise((r) => setTimeout(r, 700))
      }
      navigate('/query')
    } catch (e) {
      setFinishState('error')
      setFinishDetail(errorMessage(e))
    }
  }

  // OpenSearch is opt-in behind a compose profile, so picking it doesn't mean it's actually
  // running yet -- check before committing, so a broken retrieval step isn't the first surprise.
  async function handleFinishClick() {
    if (vectorStore === 'opensearch') {
      const reachable = await endpoints.opensearchStatus().then((s) => s.reachable).catch(() => false)
      if (!reachable) {
        setShowOsModal(true)
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
                onNext={() => setStep(needsCreds ? 'creds' : 'store')}
                computeNote={computeNote}
              />
            )}
            {step === 'creds' && (
              <StepCredentials
                choice={choice}
                creds={creds}
                onChange={setCreds}
                onBack={() => setStep('llm')}
                onNext={() => setStep('store')}
              />
            )}
            {step === 'store' && (
              <StepStore
                selected={vectorStore}
                onSelect={setVectorStore}
                onBack={() => setStep(needsCreds ? 'creds' : 'llm')}
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
    </div>
  )
}
