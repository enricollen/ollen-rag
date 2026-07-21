import { useState } from 'react'
import { motion } from 'motion/react'
import { endpoints, errorMessage } from '../../api/client'
import type { OnboardingTestRequest } from '../../api/types'
import { Button } from '../../components/Button'
import { Field, TextInput } from '../../components/Field'
import { Spinner } from '../../components/Misc'
import { CheckIcon, XIcon } from '../../components/icons'
import { toast } from '../../store/toastStore'
import { ProviderCard } from './ProviderCard'
import type { ModalityChoice } from './providers'

type TestState = 'idle' | 'testing' | 'ok' | 'fail'

// shared step for the optional embedding / reranker picks: provider cards, credential fields when
// needed, test-before-next for cloud, and a skip that leaves the local default in place.
export function StepOptionalProvider({
  title,
  subtitle,
  skipLabel,
  choices,
  selected,
  onSelect,
  providerKey,
  creds,
  onChangeCreds,
  testTarget,
  onBack,
  onNext,
  onSkip,
}: {
  title: string
  subtitle: string
  skipLabel: string
  choices: ModalityChoice[]
  selected: string
  onSelect: (id: string) => void
  /** settings key written for this modality, e.g. embedding_provider */
  providerKey: string
  creds: Record<string, string>
  onChangeCreds: (creds: Record<string, string>) => void
  testTarget: OnboardingTestRequest['target']
  onBack: () => void
  onNext: () => void
  onSkip: () => void
}) {
  const choice = choices.find((c) => c.id === selected) ?? choices[0]
  const [testState, setTestState] = useState<TestState>('idle')
  const [detail, setDetail] = useState('')

  // hide credential fields already filled in an earlier step (e.g. watsonx key from the llm pick),
  // but always show model fields so the operator can still override the default.
  const visibleFields = choice.fields.filter((f) => {
    const already = (creds[f.key] ?? '').trim()
    if (!already) return true
    return f.key.includes('model') || f.key.includes('embedding') || f.key.includes('rerank')
  })

  async function runTest() {
    setTestState('testing')
    try {
      const changes = { [providerKey]: choice.id, ...creds }
      const res = await endpoints.onboardingTest({ target: testTarget, changes })
      setTestState(res.ok ? 'ok' : 'fail')
      setDetail(res.ok ? 'Connected' : res.detail)
    } catch (e) {
      setTestState('fail')
      setDetail(errorMessage(e))
      toast(errorMessage(e), 'error')
    }
  }

  const canNext = choice.keyless || testState === 'ok'

  return (
    <div>
      <motion.h1 initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="font-display text-3xl text-ink mb-2">
        {title}
      </motion.h1>
      <motion.p initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }} className="text-ink-dim mb-8">
        {subtitle}
      </motion.p>
      <div className="flex flex-col gap-3">
        {choices.map((c, i) => (
          <motion.div
            key={c.id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08 + i * 0.04 }}
          >
            <ProviderCard
              label={c.label}
              description={c.description}
              selected={selected === c.id}
              onClick={() => {
                onSelect(c.id)
                setTestState('idle')
              }}
            />
          </motion.div>
        ))}
      </div>

      {visibleFields.length > 0 && (
        <div className="flex flex-col gap-1 mt-6">
          {visibleFields.map((f, i) => (
            <motion.div key={f.key} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 + i * 0.05 }}>
              <Field label={f.label}>
                <TextInput
                  type={f.secret ? 'password' : 'text'}
                  value={creds[f.key] ?? ''}
                  onChange={(e) => {
                    onChangeCreds({ ...creds, [f.key]: e.target.value })
                    setTestState('idle')
                  }}
                />
              </Field>
            </motion.div>
          ))}
        </div>
      )}

      {!choice.keyless && (
        <motion.div
          animate={testState === 'fail' ? { x: [0, -8, 8, -6, 6, 0] } : {}}
          transition={{ duration: 0.4 }}
          className={`flex items-center gap-2 text-sm mt-4 mb-2 min-h-[1.5rem] ${
            testState === 'ok' ? 'text-good' : testState === 'fail' ? 'text-bad' : 'text-ink-dim'
          }`}
        >
          {testState === 'testing' && (
            <>
              <Spinner /> testing connection…
            </>
          )}
          {testState === 'ok' && (
            <motion.span
              initial={{ scale: 0.6 }}
              animate={{ scale: 1 }}
              transition={{ type: 'spring', stiffness: 400 }}
              className="inline-flex items-center gap-1.5"
            >
              <CheckIcon size={14} /> {detail}
            </motion.span>
          )}
          {testState === 'fail' && (
            <span className="inline-flex items-center gap-1.5">
              <XIcon size={14} /> {detail}
            </span>
          )}
        </motion.div>
      )}

      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.35 }} className="flex items-center justify-between mt-8">
        <Button variant="ghost" onClick={onBack}>
          Back
        </Button>
        <div className="flex gap-3">
          <Button variant="ghost" onClick={onSkip}>
            {skipLabel}
          </Button>
          {!choice.keyless && (
            <Button variant="secondary" onClick={runTest} disabled={testState === 'testing'}>
              Test connection
            </Button>
          )}
          <Button variant="primary" onClick={onNext} disabled={!canNext} className="px-6">
            Next
          </Button>
        </div>
      </motion.div>
    </div>
  )
}
