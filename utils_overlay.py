import cv2

def draw_feedback_overlay(image, detector, result, action, confidence):
    h, w = image.shape[:2]
    box_width = 165
    box_height = 200
    font_scale_title = 0.4
    font_scale_body = 0.35
    line_height = 18

    # Nền dọc bên trái
    cv2.rectangle(image, (0, 0), (box_width, box_height), (30, 30, 30), -1)

    # Tiêu đề
    cv2.putText(image, f"MODE: {detector.current_exercise}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, (255, 255, 255), 1)
    cv2.putText(image, f"ACTION: {action.upper()} ({confidence:.2f})", (10, 20 + line_height),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (200, 200, 200), 1)

    # Tuỳ theo bài tập:
    if detector.current_exercise == "Bicep Curl" and result:
        l, r = result['bicep_left_analyzer'], result['bicep_right_analyzer']
        cv2.putText(image, f"L: REP={l.counter} STG={l.stage}", (10, 20 + line_height*2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (255, 255, 255), 1)
        cv2.putText(image, f"   {l.feedback}", (10, 20 + line_height*3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if l.feedback=="GOOD" else (0,0,255), 1)
        cv2.putText(image, f"R: REP={r.counter} STG={r.stage}", (10, 20 + line_height*4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (255, 255, 255), 1)
        cv2.putText(image, f"   {r.feedback}", (10, 20 + line_height*5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if r.feedback=="GOOD" else (0,0,255), 1)

    elif detector.current_exercise == "Plank" and result:
        status = result['status']
        conf = result['confidence']
        color = (0, 255, 0) if status == "Correct" else (0, 0, 255)
        cv2.putText(image, f"STATUS: {status} ({conf:.2f})", (10, 20 + line_height*2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, color, 1)

    elif detector.current_exercise == "Squat" and result:
        cv2.putText(image, f"REP={result['counter']} STG={result['stage']}", (10, 20 + line_height*2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (255,255,255), 1)
        cv2.putText(image, f"FOOT={result['foot_placement']}", (10, 20 + line_height*3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if result['foot_placement']=="Correct" else (0,0,255), 1)
        cv2.putText(image, f"KNEE={result['knee_placement']}", (10, 20 + line_height*4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if result['knee_placement']=="Correct" else (0,0,255), 1)

    elif detector.current_exercise == "Lunge" and result:
        cv2.putText(image, f"REP={result['counter']} STG={result['stage']}({result['stage_confidence']:.2f})", (10, 20 + line_height*2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (255,255,255), 1)
        cv2.putText(image, f"KNEE: {'OK' if not result['knee_angle_error'] else 'BAD'}", (10, 20 + line_height*3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if not result['knee_angle_error'] else (0,0,255), 1)
        cv2.putText(image, f"TOE: {'OK' if not result['knee_over_toe_error'] else 'OVER'}", (10, 20 + line_height*4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if not result['knee_over_toe_error'] else (0,0,255), 1)
        cv2.putText(image, f"BACK: {'OK' if not result['back_posture_error'] else 'BAD'}", (10, 20 + line_height*5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if not result['back_posture_error'] else (0,0,255), 1)

    elif detector.current_exercise == "Situp" and result:
        msg = detector.situp_analysis.get_feedback_message()
        fb_color = (0, 255, 0) if msg == "Good form" else (0, 255, 255)
        cv2.putText(image, f"REP={result['counter']} STG={result['stage'].upper()}", (10, 20 + line_height*2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (255,255,255), 1)
        cv2.putText(image, f"BACK={'OK' if not result['back_angle_error'] else 'LOW'}", (10, 20 + line_height*3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if not result['back_angle_error'] else (0,0,255), 1)
        cv2.putText(image, f"LEG={'STABLE' if not result['leg_stability_error'] else 'UNSTABLE'}", (10, 20 + line_height*4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if not result['leg_stability_error'] else (0,0,255), 1)
        cv2.putText(image, f"FB: {msg}", (10, 20 + line_height*5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, fb_color, 1)

    return image