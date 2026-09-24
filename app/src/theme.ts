/**
 * New Genre, translated to React Native.
 *
 * The web version of this system leans on a full-bleed gradient wash and a 64-72px
 * condensed serif masthead. Neither survives a 390pt phone screen intact, so two
 * things are adapted rather than copied: display type drops to 34-40pt (the same
 * monumental *proportion* against a narrower measure), and the dawn arc appears once,
 * behind the header, instead of as a page-length wash.
 *
 * Everything else is the reference values unchanged — including the rule that status
 * is never carried by hue. `hardhat` and `no-hardhat` are distinguished by stroke
 * weight and a filled versus hollow label, so the overlay still reads for the ~8% of
 * men with red-green colour blindness, and on a phone in direct sunlight.
 */

export const colors = {
  parchment: '#ffffff',
  onyx: '#0c1018',
  charredUmber: '#1e1310',
  slateVeil: '#6d7074',
  ashMist: '#9e9fa3',
  /** #f5f5f5 card fill, from the Process Card spec. */
  card: '#f5f5f5',
} as const;

/** The dawn arc, sampled at its documented stops for expo-linear-gradient. */
export const dawnArc = {
  colors: ['#280e01', '#182644', '#5a769f', '#87a1c4', '#c1d3e6', '#fef9e1', '#f7f3f0'],
  locations: [0, 0.152608, 0.30284, 0.433787, 0.588313, 0.797139, 1],
} as const;

export const space = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  xxl: 32,
} as const;

export const radius = {
  card: 16,
  pill: 50,
  small: 8,
} as const;

/**
 * The web build loads Instrument Serif and DM Sans. On Android those are bundled via
 * expo-font (see App.tsx); the string names here are the family names that registers.
 */
export const fonts = {
  display: 'InstrumentSerif',
  body: 'DMSans',
} as const;

export const type = {
  caption: { fontSize: 12, lineHeight: 14, letterSpacing: -0.12 },
  bodySm: { fontSize: 14, lineHeight: 20, letterSpacing: -0.14 },
  body: { fontSize: 16, lineHeight: 21, letterSpacing: -0.16 },
  subheading: { fontSize: 20, lineHeight: 24, letterSpacing: -0.2 },
  headingSm: { fontSize: 24, lineHeight: 28, letterSpacing: -0.24 },
  heading: { fontSize: 32, lineHeight: 35, letterSpacing: -0.32 },
  /** Phone-scaled display: the reference 64px would wrap after two words here. */
  display: { fontSize: 40, lineHeight: 42, letterSpacing: -0.8 },
} as const;

/**
 * Detection overlay styling. Deliberately monochrome, per the system's rule that no
 * status is expressed in hue: a violation is the heavier, solid, filled-label box and
 * a compliant worker is the lighter hollow one. Legible in greyscale, which is the test.
 */
export const overlay = {
  violation: { stroke: colors.onyx, strokeWidth: 3, labelFill: colors.onyx, labelText: colors.parchment },
  compliant: { stroke: colors.parchment, strokeWidth: 1.5, labelFill: 'transparent', labelText: colors.parchment },
} as const;
