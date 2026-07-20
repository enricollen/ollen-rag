import type { SVGProps } from 'react'

/*
 * Small, hand-rolled outline icon set (24x24, stroke-based, lucide-style) -- kept local so the
 * console has zero extra runtime dependency. Replaces every emoji-as-icon across the app per the
 * "no emoji icons" rule: consistent stroke width, currentColor, and sizing everywhere.
 */

export interface IconProps extends SVGProps<SVGSVGElement> {
  size?: number | string
}

function Icon({ size = 16, strokeWidth = 1.8, className = '', children, ...rest }: IconProps & { children: React.ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`inline-block flex-shrink-0 ${className}`}
      {...rest}
    >
      {children}
    </svg>
  )
}

export function GearIcon(props: IconProps) {
  return (
    <Icon {...props} strokeWidth={0} fill="currentColor">
      <path d="M19.4 13a7.5 7.5 0 0 0 .07-1 7.5 7.5 0 0 0-.07-1l2.03-1.58a.5.5 0 0 0 .12-.64l-1.92-3.32a.5.5 0 0 0-.6-.22l-2.4.96a7.4 7.4 0 0 0-1.7-1L14.5 2.6a.5.5 0 0 0-.5-.4h-3.84a.5.5 0 0 0-.5.4l-.43 2.58a7.4 7.4 0 0 0-1.7 1l-2.4-.96a.5.5 0 0 0-.6.22L2.6 8.76a.5.5 0 0 0 .12.64L4.75 11a7.5 7.5 0 0 0 0 2l-2.03 1.58a.5.5 0 0 0-.12.64l1.92 3.32c.14.24.42.33.66.22l2.4-.96c.53.43 1.1.77 1.7 1l.43 2.58c.05.24.26.4.5.4h3.84c.24 0 .45-.16.5-.4l.43-2.58c.6-.23 1.17-.57 1.7-1l2.4.96c.24.1.52 0 .66-.22l1.92-3.32a.5.5 0 0 0-.12-.64L19.4 13ZM12 15.5A3.5 3.5 0 1 1 12 8.5a3.5 3.5 0 0 1 0 7Z" />
    </Icon>
  )
}

export function PackageIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3 7.5 12 3l9 4.5v9L12 21l-9-4.5v-9Z" />
      <path d="M3 7.5 12 12l9-4.5" />
      <path d="M12 12v9" />
    </Icon>
  )
}

export function FileTextIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M6.5 2.5h8l4 4v14.5a.5.5 0 0 1-.5.5h-11.5a.5.5 0 0 1-.5-.5v-18a.5.5 0 0 1 .5-.5Z" />
      <path d="M14.5 2.5v4h4" />
      <line x1="9" y1="13" x2="15" y2="13" />
      <line x1="9" y1="17" x2="15" y2="17" />
    </Icon>
  )
}

export function AlertTriangleIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M12 3 2.5 20h19L12 3Z" />
      <line x1="12" y1="9.5" x2="12" y2="14" />
      <circle cx="12" cy="17" r="0.6" fill="currentColor" stroke="none" />
    </Icon>
  )
}

export function CheckIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <polyline points="4 12.5 9.5 18 20 6" />
    </Icon>
  )
}

export function XIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <line x1="5.5" y1="5.5" x2="18.5" y2="18.5" />
      <line x1="18.5" y1="5.5" x2="5.5" y2="18.5" />
    </Icon>
  )
}

export function LockIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <rect x="4.5" y="11" width="15" height="10" rx="2" />
      <path d="M8 11V7a4 4 0 0 1 8 0v4" />
    </Icon>
  )
}

export function BookOpenIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3 5.2c2-1.1 5-1.1 7 0v14c-2-1.1-5-1.1-7 0v-14Z" />
      <path d="M21 5.2c-2-1.1-5-1.1-7 0v14c2-1.1 5-1.1 7 0v-14Z" />
    </Icon>
  )
}

export function FolderIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3 6.5a1 1 0 0 1 1-1h4.6l1.8 2H20a1 1 0 0 1 1 1v9.5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-11.5Z" />
    </Icon>
  )
}

export function FolderOpenIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3 7a1 1 0 0 1 1-1h4.6l1.8 2H19a1 1 0 0 1 1 1v.5H8l-2.5 8H2.5L3 7Z" />
      <path d="M5.5 17.5 8 9.5h13l-2.5 8h-13Z" />
    </Icon>
  )
}

export function ScissorsIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="6.3" cy="7" r="2.2" />
      <circle cx="6.3" cy="17" r="2.2" />
      <line x1="8" y1="8.2" x2="20" y2="19" />
      <line x1="8" y1="15.8" x2="20" y2="5" />
    </Icon>
  )
}

export function PlusIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </Icon>
  )
}

export function BarChartIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <line x1="5" y1="21" x2="5" y2="10" />
      <line x1="12" y1="21" x2="12" y2="5" />
      <line x1="19" y1="21" x2="19" y2="14" />
      <line x1="3" y1="21" x2="21" y2="21" />
    </Icon>
  )
}

export function TrashIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M5 7h14" />
      <path d="M9.5 7V4.5h5V7" />
      <path d="M6.5 7 7.5 20h9l1-13" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </Icon>
  )
}

export function CpuIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <rect x="6.5" y="6.5" width="11" height="11" rx="1.5" />
      <rect x="10" y="10" width="4" height="4" />
      <line x1="9" y1="2.5" x2="9" y2="6.5" />
      <line x1="15" y1="2.5" x2="15" y2="6.5" />
      <line x1="9" y1="17.5" x2="9" y2="21.5" />
      <line x1="15" y1="17.5" x2="15" y2="21.5" />
      <line x1="2.5" y1="9" x2="6.5" y2="9" />
      <line x1="2.5" y1="15" x2="6.5" y2="15" />
      <line x1="17.5" y1="9" x2="21.5" y2="9" />
      <line x1="17.5" y1="15" x2="21.5" y2="15" />
    </Icon>
  )
}

export function SparklesIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M11.5 3 13.2 8.3 18.5 10 13.2 11.7 11.5 17 9.8 11.7 4.5 10 9.8 8.3 11.5 3Z" />
      <path d="M18.5 14 19.2 16 21.2 16.7 19.2 17.4 18.5 19.4 17.8 17.4 15.8 16.7 17.8 16 18.5 14Z" />
    </Icon>
  )
}

export function InfoIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="12" cy="12" r="9" />
      <line x1="12" y1="11" x2="12" y2="16" />
      <circle cx="12" cy="7.6" r="0.6" fill="currentColor" stroke="none" />
    </Icon>
  )
}

export function ChevronRightIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <polyline points="9 5.5 15.5 12 9 18.5" />
    </Icon>
  )
}

export function ChevronLeftIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <polyline points="15 5.5 8.5 12 15 18.5" />
    </Icon>
  )
}

export function SunIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="12" cy="12" r="4.2" />
      <line x1="12" y1="2.5" x2="12" y2="5.2" />
      <line x1="12" y1="18.8" x2="12" y2="21.5" />
      <line x1="4.4" y1="4.4" x2="6.3" y2="6.3" />
      <line x1="17.7" y1="17.7" x2="19.6" y2="19.6" />
      <line x1="2.5" y1="12" x2="5.2" y2="12" />
      <line x1="18.8" y1="12" x2="21.5" y2="12" />
      <line x1="4.4" y1="19.6" x2="6.3" y2="17.7" />
      <line x1="17.7" y1="6.3" x2="19.6" y2="4.4" />
    </Icon>
  )
}

export function MoonIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5Z" />
    </Icon>
  )
}

export function MessageSquareIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 5.5a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H9.5L5 20V16.5H6a2 2 0 0 1-2-2v-9Z" />
    </Icon>
  )
}

export function TargetIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="12" cy="12" r="8.2" />
      <circle cx="12" cy="12" r="4.6" />
      <circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" />
    </Icon>
  )
}

export function DatabaseIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <ellipse cx="12" cy="5.2" rx="7.8" ry="2.7" />
      <path d="M4.2 5.2v13.6c0 1.5 3.5 2.7 7.8 2.7s7.8-1.2 7.8-2.7V5.2" />
      <path d="M4.2 12c0 1.5 3.5 2.7 7.8 2.7s7.8-1.2 7.8-2.7" />
    </Icon>
  )
}

export function UploadIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M12 15.5V4" />
      <polyline points="7.5 8.5 12 4 16.5 8.5" />
      <path d="M4.5 15.5v3a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-3" />
    </Icon>
  )
}

export function TagIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M12.5 3.5H6a1 1 0 0 0-1 1V11a1 1 0 0 0 .3.7l9 9a1 1 0 0 0 1.4 0l6-6a1 1 0 0 0 0-1.4l-9-9a1 1 0 0 0-.2-.4Z" />
      <circle cx="8.2" cy="7.7" r="1.1" fill="currentColor" stroke="none" />
    </Icon>
  )
}

export function SearchIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="10.5" cy="10.5" r="6.5" />
      <line x1="15.3" y1="15.3" x2="20.5" y2="20.5" />
    </Icon>
  )
}

export function LayersIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M12 3 21 8l-9 5-9-5 9-5Z" />
      <path d="M3 12l9 5 9-5" />
      <path d="M3 16l9 5 9-5" />
    </Icon>
  )
}

export function CopyIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <rect x="8.5" y="8.5" width="12" height="12" rx="1.5" />
      <path d="M15.5 8.5V5.5a1.5 1.5 0 0 0-1.5-1.5H5.5A1.5 1.5 0 0 0 4 5.5v9A1.5 1.5 0 0 0 5.5 16H8" />
    </Icon>
  )
}

export function TerminalIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <rect x="2.5" y="4" width="19" height="16" rx="2" />
      <polyline points="6.5 9.5 10.5 12.5 6.5 15.5" />
      <line x1="13" y1="15.5" x2="17.5" y2="15.5" />
    </Icon>
  )
}
