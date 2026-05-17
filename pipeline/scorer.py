from dataclasses import dataclass
from typing import Dict,Tuple


@dataclass
class Score:
    factuality: float
    confidence: float
    completeness: float
    consistency: float
    reasoning: float

    @property

    def weighted_total(self) -> float:
        weights={
            "factuality": 0.35,
            "confidence": 0.15,
            "completeness": 0.20,
            "consistency": 0.15,
            "reasoning": 0.15,
        }

        return (
            self.factuality * weights["factuality"]+self.confidence * weights["confidence"]+ self.completeness*weights["completeness"]+self.consistency*weights["consistency"]+self.reasoning*weights["reasoning"]
        )
    @property
    def confidence_label(self) -> str:
        total=self.weighted_total
        if total>=0.80:
            return "High Confidence"
        elif total>=0.60:
            return "Medium Confidence"
        elif total>=0.40:
            return "Low Confidence"
        else:
            return "Needs Verification"
    

    def to_dict(self) -> Dict:
        return {
            "factuality": self.factuality,
            "confidence": self.confidence,
            "completeness": self.completeness,
            "consistency": self.consistency,
            "reasoning": self.reasoning,
            "weighted_total": self.weighted_total,
            "label": self.confidence_label,
        }
    

def parse_critique_response(text: str) -> Tuple[Score,str]:
    import json
    import re

    defaults={"factuality": 0.5, "confidence": 0.5, "completeness": 0.5, "consistency": 0.5, "reasoning": 0.5}

    feedback="Failed to parse FeedBack"

    json_str=text

    match=re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)


    if match:
        json_str=match.group(1).strip()

    else:
        start= text.find('{')
        end=text.rfind('}')
        if start !=-1 and end !=-1:
            json_str=text[start:end+1]


    try:
        data=json.loads(json_str)
        for dim in defaults:
            if dim in data:
                val=float(data[dim])
                defaults[dim]=min(max(val/10.0 if val>1 else val,0.0),1.0)

            if "feedback" in data:
                feedback=str(data["feedback"])
    except (json.JSONDecodeError,ValueError) as e:
        print(f"Failed to parse critique JSON: {e}\nRaw text: {text}")
    return Score(**defaults), feedback