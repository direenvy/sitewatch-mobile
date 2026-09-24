/**
 * The boxes, drawn over the photo.
 *
 * Detections come back in the *original* photo's pixel space, which is almost never
 * the size the photo is displayed at. Rather than rescaling every box, the whole SVG
 * is given a viewBox in image coordinates and stretched to the display rect — so the
 * numbers drawn here are the numbers the detector produced, and there is one less
 * place for a coordinate bug to hide.
 *
 * The cost is that stroke widths scale too, which is why they are divided back out.
 */

import React from 'react';
import Svg, { G, Rect, Text as SvgText } from 'react-native-svg';

import { Detection } from '../detect';
import { colors, fonts, overlay } from '../theme';

interface Props {
  detections: Detection[];
  /** Original photo dimensions — the SVG's coordinate system. */
  imageWidth: number;
  imageHeight: number;
  /** On-screen size of the photo, in points. */
  displayWidth: number;
  displayHeight: number;
}

export function Overlay({ detections, imageWidth, imageHeight, displayWidth, displayHeight }: Props) {
  // One scale factor for both axes: the preview uses resizeMode="contain", so the
  // photo keeps its aspect ratio and the ratio is the same either way.
  const k = imageWidth / displayWidth;
  const labelSize = 11 * k;

  return (
    <Svg
      width={displayWidth}
      height={displayHeight}
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      style={{ position: 'absolute', top: 0, left: 0 }}
    >
      {detections.map((d, i) => {
        const style = d.cls === 'no-hardhat' ? overlay.violation : overlay.compliant;
        const w = d.x2 - d.x1;
        const h = d.y2 - d.y1;
        const label = `${d.cls === 'no-hardhat' ? 'NO HARD HAT' : 'HARD HAT'} ${Math.round(d.score * 100)}`;
        const labelW = label.length * labelSize * 0.58 + 6 * k;
        // Flip the label inside the box when the detection sits against the top edge,
        // which is exactly where heads tend to be.
        const above = d.y1 - labelSize * 1.5 > 0;
        const labelY = above ? d.y1 - labelSize * 1.5 : d.y1;

        return (
          <G key={i}>
            {/* A dark halo under every stroke: a white box vanishes on concrete and a
                dark one vanishes on a shadow, so each gets an opposite-tone outline. */}
            <Rect
              x={d.x1}
              y={d.y1}
              width={w}
              height={h}
              fill="none"
              stroke={d.cls === 'no-hardhat' ? colors.parchment : colors.onyx}
              strokeWidth={style.strokeWidth * k + 2 * k}
              strokeOpacity={0.35}
            />
            <Rect
              x={d.x1}
              y={d.y1}
              width={w}
              height={h}
              fill="none"
              stroke={style.stroke}
              strokeWidth={style.strokeWidth * k}
            />
            {style.labelFill !== 'transparent' && (
              <Rect x={d.x1} y={labelY} width={labelW} height={labelSize * 1.5} fill={style.labelFill} />
            )}
            <SvgText
              x={d.x1 + 3 * k}
              y={labelY + labelSize * 1.1}
              fill={style.labelText}
              fontSize={labelSize}
              fontFamily={fonts.body}
              stroke={style.labelFill === 'transparent' ? colors.onyx : 'none'}
              strokeWidth={style.labelFill === 'transparent' ? 0.7 * k : 0}
              // Paint the halo behind the glyph rather than over it.
              paintOrder="stroke"
            >
              {label}
            </SvgText>
          </G>
        );
      })}
    </Svg>
  );
}
