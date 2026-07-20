import { useState } from 'react'
import { motion } from 'motion/react'
import { endpoints, errorMessage } from '../../api/client'
import { Button } from '../../components/Button'
import { Field, TextInput } from '../../components/Field'
import { Spinner } from '../../components/Misc'
import { CheckIcon, XIcon } from '../../components/icons'
import { toast } from '../../store/toastStore'
import type { LlmChoice } from './providers'

type TestState = 'idle' | 'testing' | 'ok' | 'fail'

export function StepCredentials({
  choice,
  creds,
  onChange,
  onBack,
  onNext,
}: {
  choice: LlmChoice
  creds: Record<string, string>
  onChange: (creds: Record<string, string>) => void
  onBack: () => void
  onNext: () => void
}) {
  const [testState, setTestState] = useState<TestState>('idle')
  const [detail, setDetail] = useState('')

  async function runTest() {
    setTestState('testing')
    try {
      const changes = { llm_provider: choice.id, ...creds }
      const res = await endpoints.onboardingTest({ target: 'llm', changes })
      setTestState(res.ok ? 'ok' : 'fail')
      setDetail(res.ok ? 'Connected' : res.detail)
    } catch (e) {
      setTestState('fail')
      setDetail(errorMessage(e))
      toast(errorMessage(e), 'error')
    }
  }

  return (
    <div>
      <motion.h1 initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="font-display text-3xl text-ink mb-2">
        {choice.label} credentials
      </motion.h1>
      <motion.p initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }} className="text-ink-dim mb-8">
        Enter your credentials, then test the connection before continuing.
      </motion.p>
      <div className="flex flex-col gap-1">
        {choice.fields.map((f, i) => (
          <motion.div key={f.key} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 + i * 0.06 }}>
            <Field label={f.label}>
              <TextInput
                type={f.secret ? 'password' : 'text'}
                value={creds[f.key] ?? ''}
                onChange={(e) => {
                  onChange({ ...creds, [f.key]: e.target.value })
                  setTestState('idle')
                }}
              />
            </Field>
          </motion.div>
        ))}
      </div>

      <motion.div
        animate={testState === 'fail' ? { x: [0, -8, 8, -6, 6, 0] } : {}}
        transition={{ duration: 0.4 }}
        className={`flex items-center gap-2 text-sm mb-4 min-h-[1.5rem] ${
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

      <div className="flex items-center justify-between mt-6">
        <Button variant="ghost" onClick={onBack}>
          Back
        </Button>
        <div className="flex gap-3">
          <Button variant="secondary" onClick={runTest} disabled={testState === 'testing'}>
            Test connection
          </Button>
          <Button variant="primary" onClick={onNext} disabled={testState !== 'ok'} className="px-6">
            Next
          </Button>
        </div>
      </div>
    </div>
  )
}
