export type TopicIcon = 'bag' | 'card' | 'ticket' | 'refund' | 'star' | 'assist' | 'passport'

export interface Topic {
  title: string
  icon: TopicIcon
  questions: string[]
}

export const TOPICS: Topic[] = [
  {
    title: 'Baggage',
    icon: 'bag',
    questions: [
      'What is the checked baggage allowance on international Economy flights?',
      'Can I bring sports equipment on my flight?',
      'What should I do if my baggage is delayed or damaged?',
    ],
  },
  {
    title: 'Booking & payment',
    icon: 'card',
    questions: [
      'What payment methods are accepted for bookings?',
      'How do I add a passenger name correction to my booking?',
      'Can I hold a booking before paying?',
    ],
  },
  {
    title: 'Check-in',
    icon: 'ticket',
    questions: [
      'When does online check-in open?',
      'How early should I arrive at the airport?',
      'Can I check in online with an infant?',
    ],
  },
  {
    title: 'Changes & refunds',
    icon: 'refund',
    questions: [
      'How do I request a refund for my ticket?',
      'Can I change the date of my flight?',
      'What happens if my flight is cancelled?',
    ],
  },
  {
    title: 'Enrich',
    icon: 'star',
    questions: [
      'How do I earn Enrich points?',
      'How can I redeem Enrich points for a flight?',
      'How long are Enrich points valid?',
    ],
  },
  {
    title: 'Special assistance',
    icon: 'assist',
    questions: [
      'How do I request wheelchair assistance?',
      'Can I fly while pregnant?',
      'Can children travel alone?',
    ],
  },
  {
    title: 'Travel documents',
    icon: 'passport',
    questions: [
      'What travel documents do I need to check in?',
      'Does my passport need to be valid for six months?',
    ],
  },
]
