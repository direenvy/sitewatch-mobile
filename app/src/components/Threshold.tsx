/**
 * The confidence slider.
 *
 * This is the one control on the screen, and it exists because the threshold is the
 * product decision Sitewatch is actually about. The web build made you move it to see
 * the trade; the phone build keeps it for the same reason, and re-runs only the decode
 * and suppression — never the network — so dragging it is instant.
 *
 * `react-native-community/slider` rather than a gesture-handler implementation: it
 * renders the platform control, which means it inherits Android's own touch target
 * and accessibility behaviour instead of approximating them.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Slider from '@react-native-community/slider';

import { DEFAULT_CONF } from '../detect';
import { colors, fonts, space, type } from '../theme';

interface Props {
  value: number;
  onChange: (v: number) => void;
}

export function Threshold({ value, onChange }: Props) {
  return (
    <View style={styles.wrap}>
      <View style={styles.row}>
        <Text style={styles.label}>Confidence threshold</Text>
        <Text style={styles.value}>{value.toFixed(2)}</Text>
      </View>
      <Slider
        value={value}
        onValueChange={onChange}
        minimumValue={0.05}
        maximumValue={0.9}
        step={0.01}
        minimumTrackTintColor={colors.onyx}
        maximumTrackTintColor={colors.ashMist}
        thumbTintColor={colors.onyx}
        style={styles.slider}
      />
      <Text style={styles.hint}>
        {value === DEFAULT_CONF
          ? `Shipped operating point — chosen by F1 on no-hardhat alone, not by mAP.`
          : value < DEFAULT_CONF
            ? `Below the shipped ${DEFAULT_CONF.toFixed(2)}: catches more violations, at the cost of false alarms.`
            : `Above the shipped ${DEFAULT_CONF.toFixed(2)}: fewer false alarms, and more violations missed.`}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    backgroundColor: colors.card,
    borderRadius: 16,
    padding: space.lg,
    gap: space.xs,
  },
  row: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline' },
  label: { ...type.bodySm, fontFamily: fonts.body, color: colors.onyx },
  value: { ...type.subheading, fontFamily: fonts.body, color: colors.onyx },
  // Negative margin pulls the platform slider's own padding back to the card edge.
  slider: { marginHorizontal: -space.sm },
  hint: { ...type.caption, lineHeight: 16, fontFamily: fonts.body, color: colors.slateVeil },
});
